import os;
import matplotlib.pyplot as plt;
import cv2;
import json;

targetDir = "lap_01";

images = os.scandir(targetDir);
labelsFile = open("labels.json", "r");
labels = json.loads(labelsFile.read());

if not(targetDir in labels.keys()):
    labels[targetDir] = {};

batchN = int(len(labels[targetDir]) / 20) + 1;
if len(labels[targetDir]) / 20 < 1: 
    batchN = 0;
nameWithoutPrefix = "_cam-image_array_.jpg";

# for c in range(batchN*20, (batchN+1)*20):
#     fig = plt.figure()
#     i = cv2.imread(targetDir + "/" + str(c) + nameWithoutPrefix);
#     if i is not None:
#         plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB));
#         plt.draw()
#         plt.pause(0.001)
#         yn = input("Ligne de stop ? - " + str(c));
#         labels[targetDir][str(c) + nameWithoutPrefix] = 1 if yn == "y" else 0;

for c in range(1760, 1900):
    labels[targetDir][str(c) + nameWithoutPrefix] = 0;

# print(labels)
labelsJSON = json.dumps(labels);

f = open("labels.json", "w")
f.write(labelsJSON)
f.close()
