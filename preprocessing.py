import os
import shutil
import matplotlib.pyplot as plt
import cv2
import numpy as np

# from supervise.pilot_train import DATASET_DIR

def cropY(img, px_from_top):
    if len(np.shape(img)) == 3:
        return img[px_from_top:np.shape(img)[0], :, :]
    return img[px_from_top:np.shape(img)[0], :]

def gaussian_filter(img, ksize=(3,3), sigmaX = 0):
    return cv2.GaussianBlur(img, ksize=ksize, sigmaX=sigmaX)

def threshold_filter(img, thr = 195):
    return cv2.threshold(img, thr, 255, type=cv2.THRESH_BINARY)

def canny(img):
    return cv2.Canny(img,100,200)

if __name__ == "__main__":
    IMAGE = cv2.imread('')
    IMAGE_BNW = cv2.imread('/Users/IMT_Atlantique/project_ia/cars/mysim/data/images/119_cam_image_array_.jpg', cv2.IMREAD_GRAYSCALE)

    DATASET_DIR_SOURCE = "/Users/IMT_Atlantique/project_ia/ia_racing_imt/supervise/dataset_drive/"
    DATASET_DIR_TARGET = "/Users/IMT_Atlantique/project_ia/ia_racing_imt/supervise/dataset_drive/preprocessed/"

    AFFICHAGE = False
    COLS = 3
    images = {
        'Original' : IMAGE,
    }

    images['IMAGE_BNW'] = gaussian_filter(IMAGE_BNW)
    images['Cropped'] = cropY(IMAGE, 40)
    images['Cropped BW'] = cropY(images['IMAGE_BNW'], 45)
    images['Cropped Gaussien'] = gaussian_filter(images['Cropped'])
    images['Threshold'] = threshold_filter(cv2.cvtColor(images['Cropped Gaussien'], cv2.COLOR_BGR2GRAY))[1]
    # images['Threshold BW'] = threshold_filter(images['Cropped BW'])[1]
    images['Canny'] = canny(images['Cropped Gaussien'])

    transfos = images.keys()


    if AFFICHAGE:
        ROWS = int(len(images)/COLS) + 1
        fig = plt.figure()
        c = 0
        for img in transfos:
            c += 1
            fig.add_subplot(ROWS, COLS, c)
            try:
                plt.imshow(cv2.cvtColor(images[img], cv2.COLOR_BGR2RGB))
            except:
                pass
            try: 
                plt.imshow(images[img], cmap='gray')
            except:
                pass
            plt.axis("off")
            plt.title(img)

        plt.show()


    images_to_preprocess = os.scandir(DATASET_DIR_SOURCE)
    for file in images_to_preprocess:
        if "jpg" in file.name:
            img = cv2.imread(file.path, cv2.IMREAD_GRAYSCALE)
            img = threshold_filter(gaussian_filter(cropY(img, 50)))[1]
            print(np.shape(img))
            plt.imsave(DATASET_DIR_TARGET + file.name, img, cmap='gray')
        elif ".json" in file.name:
            shutil.copy(file.path, DATASET_DIR_TARGET + file.name)