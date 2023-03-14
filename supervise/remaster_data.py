__author__ = "Amaury COLIN"
__credits__ = ["Amaury COLIN", "Malick BA", "Jhon MUNOZ"]
__date__ = "2023.03.14"
__version__ = "0.0.1"

'''
    Fichier pour fusionner des tubs entre eux pour entrainer un modele avec differents preprocessings
    
    Structure de l'enregistrement :
        target_tub/
        |
        |-- raw/
        |   |-- images/
        |   |   |-- img1
        |   |   |-- …
        |   |-- catalog1
        |   |-- catalog…
        |   |-- manifest.json
        |
        |-- prepro1/
        |   |-- images/
        |   |   |-- img1
        |   |   |-- …
        |   |-- catalog1
        |   |-- catalog…
        |   |-- manifest.json
        |
    …
'''

############### IMPORTS ###############

import argparse # Arguments
import json # Utilisation du json
import math # Arrondis
import os # Chemins
import shutil # Copie de fichiers
import cv2 # Preprocessings
import numpy as np # La base

############### PARAMETRES ###############

TUBS_MASTER = '/Users/IMT_Atlantique/project_ia/data_all'
CROP = 40
PREPRO = ['lines', 'bnw'] # Prepro existant : lines|bnw
TARGET_DIR = os.path.join(os.getcwd(), "test_tub")


############### Classes utiles ###############

class Preprocess():
    
    ''' Classe pour preprocesser les images '''
    
    def __init__(self, dir, image, crop, method):
        self.img = cv2.imread(os.path.join(dir,image))
        self.img_name = image
        self.cropY(crop)
        if method == "lines":
            self.lines()
        elif method == "bnw":
            self.bnw()
        else:
            pass
    
    def cropY(self, crop):
        if len(np.shape(self.img)) == 3:
            self.img = self.img[crop:np.shape(self.img)[0], :, :]
        self.img[crop:np.shape(self.img)[0], :]
    
    def gaussian(self, ksize=(3,3), sigmaX = 0):
        self.img = cv2.GaussianBlur(self.img, ksize=ksize, sigmaX=sigmaX)
    
    def lines(self):
        self.gaussian()
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 150, 200,apertureSize = 3)
        minLineLength = 20
        maxLineGap = 5
        lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
        
        img = np.zeros((img.shape[0], img.shape[1], 3), dtype = "uint8")
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
                pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
                cv2.polylines(img, [pts], True, (0,0,255), 3)
        img[54:, 33:128, :] = 0 # Masque pour le parchoc
        self.img = img
    
    def bnw(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    
    def save(self, path):
        cv2.imwrite(path, self.img)
        


class TubManager():
    
    ''' Classe pour faciliter la lecture '''
    
    tub_dir = None
    images_dir = None
    catalogs = []
    catalogs_manifests = []
    catalog_values = None
    line_lengths_values = None
    images = []
    manifest_file = None
    manifest = None
    
    def __init__(self, tub_dir = None):
        self.tub_dir = tub_dir
        self.images_dir = None
        self.catalogs = []
        self.catalogs_manifests = []
        self.catalog_values = None
        self.line_lengths_values = None
        self.images = []
        self.manifest_file = None
        self.manifest = None
        if self.tub_dir:
            self.images_dir = os.path.join(tub_dir, 'images')
            for f in os.scandir(self.tub_dir):
                if "catalog" in f.name:
                    if "manifest" in f.name:
                        self.catalogs_manifests.append(os.path.join(tub_dir, f.name))
                    else:
                        self.catalogs.append(os.path.join(tub_dir, f.name))
            self.manifest_file = os.path.join(tub_dir, 'manifest.json')
            
    def get_manifest(self):
        if self.manifest_file:
            j = open(self.manifest_file, 'r')
            self.manifest = json.loads(j.read())
            return self.manifest
        return None
    
    def get_values(self):
        if len(self.catalogs) != 0:
            catalog_values = []
            for c in self.catalogs:
                f = open(c, 'r')
                line = f.readline()
                while line:
                    catalog_values.append(json.loads(line))
                    line = f.readline()
                f.close()
            self.catalog_values = catalog_values
            return catalog_values
        return None
    
    def get_line_lengths(self):
        if len(self.catalogs_manifests) != 0:
            line_lengths_values = []
            for c in self.catalogs_manifests:
                f = open(c, 'r')
                line_lengths_values.append(json.loads(f.read())['line_lengths'])
                f.close()
            self.line_lengths_values = line_lengths_values
            return line_lengths_values
        return None
    
    def get_images(self):
        if self.catalog_values:
            images = []
            for c in self.catalog_values:
                img_name = c['cam/image_array']
                img_path_name = self.tub_dir
                images.append({"dir": img_path_name, "img": img_name})
            self.images = images
            return images
        return None

    def set_images(self, images):
        self.images = images

    def set_catalog_values(self, cv):
        self.catalog_values = cv
    
    def set_line_lengths_values(self, llv):
        self.line_lengths_values = llv

    def create_catalogs(self, path, catalog_size = 1000):
        if self.catalog_values and self.line_lengths_values:
            L = len(self.catalog_values)
            N_catalogs = math.ceil(L/catalog_size)
            
            if L/catalog_size != N_catalogs:
                N_catalogs += 1
            
            for c in range(1, N_catalogs):
                ll = []
                with open(os.path.join(path, f"catalog_{c - 1}.catalog"), "w") as new_catalog:
                    for i in range(catalog_size*(c - 1), catalog_size*c):
                        if i < len(self.catalog_values):
                            v = self.catalog_values[i]
                            v["_index"] = str(i)
                            v["cam/image_array"] = v["_index"] + "_cam_image_array_.jpg"

                            jv = json.dumps(v) + '\n'
                            ll.append(len(jv))
                            new_catalog.write(jv)
                with open(os.path.join(path, f"catalog_{c - 1}.catalog_manifest"), "w") as new_catalog_manifest:
                    jm = json.dumps({"created_at": 1678368955.0940523, "line_lengths": ll, "path": f"catalog_{c - 1}.catalog_manifest", "start_index": catalog_size*(c - 1)})
                    new_catalog_manifest.write(jm)
            
            return N_catalogs
        pass
    
    def move_images(self, path, crop = 40, preprocessing = None):
        if self.images:
            if not os.path.isdir(os.path.join(path, 'images')):
                os.mkdir(os.path.join(path, 'images'))
            for i in range(len(self.images)):
                new_image_name = str(i) + "_cam_image_array_.jpg"
                if preprocessing is None:
                    shutil.copy(os.path.join(self.images[i]['dir'], 'images', self.images[i]['img']), os.path.join(path, 'images', new_image_name))
                else:
                    (Preprocess(os.path.join(self.images[i]['dir'], 'images'), self.images[i]['img'], crop, preprocessing)).save(os.path.join(path, 'images', new_image_name))
            
    
    def create_manifest(self, path, N_catalogs, catalog_size = 1000):
        if self.catalog_values and self.line_lengths_values:
            with open(os.path.join(path, "manifest.json"), "w") as new_manifest:
                new_manifest.write(json.dumps(["cam/image_array", "user/angle", "user/throttle", "user/mode"]) + "\n")
                new_manifest.write(json.dumps(["image_array", "float", "float", "str"]) + "\n")
                new_manifest.write(json.dumps({}) + "\n")
                new_manifest.write(json.dumps({"created_at": 1678368870.8656437, "sessions": {"all_full_ids": ["23-03-09_0"], "last_id": 0, "last_full_id": "23-03-09_0"}}) + "\n")
                new_manifest.write(json.dumps({"paths": [f"catalog_{i}.catalog" for i in range(N_catalogs - 1)], "current_index": len(self.catalog_values), "max_len": catalog_size, "deleted_indexes": []}) + "\n")
            return True
        return False
    
    
    def save(self, tub_dir, tub_name, catalog_size = 1000, crop = 40, preprocessing = None):
        path = os.path.join(tub_dir, tub_name)
        if not os.path.isdir(path):
            os.mkdir(path)
            
        Ncatalogs = self.create_catalogs(path, catalog_size)
        self.create_manifest(path, Ncatalogs, catalog_size)
        self.move_images(path, crop, preprocessing)
            
############### EXECUTION DU FICHIER ###############

if __name__ == "__main__":
    
    ## Recuperation des arguments ##
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--from_dir", help="Dossier avec les tubs a fusionner")
    argParser.add_argument("-c", "--crop", help="Nombre de pixels a rogner par rapport au haut")
    argParser.add_argument("-p", "--preprocessing", nargs="*", help="Liste des preprocessings a utiliser")
    argParser.add_argument("-t", "--target", help="Chemin de destination")

    args = argParser.parse_args()

    tubs_main_dir = TUBS_MASTER if args.from_dir is None else args.from_dir
    cropPx = CROP if args.crop is None else args.crop
    prepros = PREPRO if args.preprocessing is None else args.preprocessing
    target_tub = TARGET_DIR if args.target is None else args.target

    ## Parcours des tubs pour la fusion ##
    
    catalog_values = []
    line_lengths = []
    images = []
    tubs = os.scandir(tubs_main_dir)
    for tub in tubs:
        tub_path = os.path.join(tubs_main_dir, tub.name)
        if tub.is_dir():
            t = TubManager(tub_path)
            catalog_values.extend(t.get_values())
            line_lengths.extend(t.get_line_lengths())
            images.extend(t.get_images())
    
    ## Creation du dossier de destination ##
    
    if not os.path.isdir(target_tub):
        os.mkdir(target_tub)
        
    ## Parametrage pour le nouveau tub ##
    
    t = TubManager()
    t.set_images(images)
    t.set_catalog_values(catalog_values)
    t.set_line_lengths_values(line_lengths)

    print("************************************************")
    print("********* Enregistrement images brutes *********")
    
    ## Enregistrement des images brutes ##
    m = t.save(target_tub, 'raw', 1000)
    
    ## Enregistrement des images rognees ##
    if 'crop' not in prepros:
        print(f"********* Enregistrement images rognees *********")
        m = t.save(target_tub, "crop", 1000, crop=cropPx, preprocessing="Crop")
    
    ## Enregistrement des images preprocessees ##
    for p in prepros:
        print(f"********* Enregistrement images {p} *********")
        m = t.save(target_tub, p, 1000, cropPx, p)

    print("************************************************")