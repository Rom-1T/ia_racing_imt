import os, shutil, json

datasetsDir = 'mydataset'
datasets = ['alt_03_001', 'dakou', 'gen_track_user_drv_right_lane', 'lap_01']

######################################
## Vider les dossiers train et test ##
######################################

dirs = ['test', 'train']
for dir in dirs:
    for filename in os.listdir(datasetsDir + '/' + dir):
        file_path = os.path.join(datasetsDir + '/' + dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            

#########################################
## Choisir le dataset de train et test ##
#########################################

moveDatasets = {
    'train': ['alt_03_001', 'dakou', 'gen_track_user_drv_right_lane'],
    'test': ['lap_01']
}


labelsFile = open("labels.json", "r");
labels = json.loads(labelsFile.read());

for dir in dirs:
    dirLabels = {}
    os.mkdir(datasetsDir+'/'+dir+'/class_0')
    os.mkdir(datasetsDir+'/'+dir+'/class_1')

    if dir in moveDatasets.keys():
        for dataset in moveDatasets[dir]:
            if dataset in datasets:
                for img in labels[dataset].keys():
                    if labels[dataset][img] == 1:
                        imgName = dataset+'_'+img
                        shutil.copyfile(dataset + '/' + img, datasetsDir+'/'+dir+'/class_1/'+imgName)
                    elif labels[dataset][img] == 0: 
                        imgName = dataset+'_'+img
                        shutil.copyfile(dataset + '/' + img, datasetsDir+'/'+dir+'/class_0/'+imgName)
                    dirLabels[imgName] = labels[dataset][img]
                    
                    
        labelsFile = open(datasetsDir+'/'+dir + "/labels.json", "w")
        labelsFile.write(json.dumps(dirLabels))
        labelsFile.close()