import os, shutil, json

# datasetsDir = './stop/new_dataset/train'
dossier_a_rapatrier = './stop/validation_dataset_sigma/'
dirs = []
labels = []

def rapatrier(dossier, labs, value):
    d = dossier + "class_" + str(value) + "/"
    
    fs = os.scandir(d)
    c = len(labs)
    for f in fs:
        c+= 1
        if '.jpg' in f.name:
            shutil.copyfile(d + f.name, dossier + f.name)
            labels.append({'img_name': f.name, 'label_value': value})
    
    return labs
            
labels = rapatrier(dossier_a_rapatrier, labels, 0)
labels = rapatrier(dossier_a_rapatrier, labels, 1)

f = open(dossier_a_rapatrier + "labels.json", 'w')
f.write(json.dumps(labels))
f.close()