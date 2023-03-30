from distutils.sysconfig import PREFIX
import json
import os
import shutil

TO_CLASSIFY_DIR = os.getcwd() + '/supervise/dataset_drive2/a_classer'
prefix = ""
LABELS_FILE_FILENAME = os.getcwd() + '/supervise/dataset_drive2/labels.json'

labels_file = open(LABELS_FILE_FILENAME, 'r')

labels = json.loads(labels_file.read())

labels_file.close()

def classify(dir, prefix):
    
    files_to_classify = os.scandir(dir)
    for file in files_to_classify:
        if file.is_dir():
            print('dir')
            classify(dir + "/" + file.name, file.name)
            shutil.rmtree(dir + "/" + file.name)
        elif file.is_file():
            if "jpg" in file.name:
                shutil.move(dir + "/" + file.name, TO_CLASSIFY_DIR + "/../" + prefix + "-" + file.name)
            elif "record" in file.name or "catalog" in file.name:
            # if "record" in file.name or "catalog" in file.name:
                data = open(file.path, "r")
                contentAll = json.loads(data.read())
                data.close()
                
                for content in contentAll:
                    if len(prefix) != 0:
                        labels[prefix + "-" + content['cam/image_array']] = {'user/throttle': content['user/throttle'], 'user/angle': content['user/angle']}
                    else:
                        labels[content['cam/image_array']] = {'user/throttle': content['user/throttle'], 'user/angle': content['user/angle']}

if __name__ == "__main__":
    classify(TO_CLASSIFY_DIR, prefix)

    labels_stringified = json.dumps(labels)

    print(labels)

    labels_file = open(LABELS_FILE_FILENAME, 'w')

    labels_file.write(labels_stringified)
    labels_file.close()