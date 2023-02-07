import json
import math
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import cv2
from alive_progress import alive_bar 

DATASET_DIR = 'supervise/dataset_drive2/'

LEARNING_RATE=0.002
WIDTH = 160
HEIGHT = 120
CHANNELS = 3 # à quoi ça sert ça ?
BATCH_SIZE = 10
NUMBER_OF_IMAGES = 5000
NUMBER_OF_BATCHES = math.floor(NUMBER_OF_IMAGES / BATCH_SIZE)
NUMBER_OF_EPOCHS = 40

TRAIN_WITH_PREPRO = False
if TRAIN_WITH_PREPRO:
    DATASET_DIR += "preprocessed/"

f = open(DATASET_DIR + 'labels.json')
labels = json.loads(f.read())

dataset_images = list(labels.keys())

#########################################################
#      On récupère l'image et le label en question      #
#########################################################

def get_images_for_new_batch(batch_number, image_in_batch):
    imgN = batch_number*BATCH_SIZE + image_in_batch
    img_arr = cv2.imread(DATASET_DIR+dataset_images[imgN], cv2.IMREAD_GRAYSCALE)
    # print("get", np.shape(img_arr))
    label = labels[dataset_images[imgN]]
    
    return camera_image_with_throttle_and_steering(img_arr, label)



################################################
#      On transforme une image en torseur      #
################################################

def camera_image_with_throttle_and_steering(camera_img_arr, label): 
    # print(np.shape(camera_img_arr))
    transformImg=tf.Compose([ tf.ToPILImage(),
                             tf.ToTensor()]) 
    camera_image_tensor = transformImg(camera_img_arr) # Transform to pytorch
    # print(np.shape(camera_image_tensor))
    return camera_image_tensor, label['user/throttle'], label['user/angle']



#######################################
#      On crée un batch d'images      #
#######################################

def LoadBatch(batch_number): # Load batch of images
    images = torch.zeros([BATCH_SIZE,CHANNELS, HEIGHT, WIDTH])
    labls = torch.zeros([BATCH_SIZE, 2])
    # print(labls)
    for i in range(BATCH_SIZE):
        images[i], labls[i][0], labls[i][1] = get_images_for_new_batch(batch_number, i)
    return images, labls

##################################
#      Définition du modèle      #
##################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # Set device GPU or CPU where the training will take place
Net = torchvision.models.resnet18(pretrained=True) # Load net
Net.fc = torch.nn.Linear(in_features=Net.fc.in_features, out_features=2, bias=True) # Change final layer to predict one value
Net = Net.to(device)
optimizer = torch.optim.SGD(params=Net.parameters(),lr=LEARNING_RATE) # Create adam optimizer
criterion = torch.nn.MSELoss()



####################################
#      Entrainement du modèle      #
####################################

for i in range(1, NUMBER_OF_EPOCHS + 1):
    print("Epoch", str(i)+"/"+str(NUMBER_OF_EPOCHS))
    bar_total = NUMBER_OF_BATCHES
    with alive_bar(bar_total) as bar:
        for batch_N in range(0, NUMBER_OF_BATCHES): # Training loop
            images, labls = LoadBatch(batch_N) # Load taining batch
            # np.shape("imgs", images)
            images = torch.autograd.Variable(images, requires_grad=False).to(device) # Load image
            # np.shape("imgsVAr", images)
            labls = torch.autograd.Variable(labls, requires_grad=False).to(device) # Load Ground truth fill level
            #    angles = torch.autograd.Variable(angles, requires_grad=False).to(device) # Load Ground truth fill level
            outputs = Net(images) # make prediction
            Net.zero_grad()
            #    print(outputs, labls)
            #    print(PredLevel, np.shape(PredLevel))
            #    print(GTFillLevel, np.shape(GTFillLevel))
            Loss = criterion(outputs, labls)
            Loss.backward()
            optimizer.step() # Apply gradient descent change to weight
            print(str(i) + "." + str(batch_N + 1),") Loss=",Loss.data.cpu().numpy()) # Display loss
            if batch_N % 500 == 499: # Save model
                    print("Saving Model" +str(batch_N + 1) + ".torch") #Save model weight
                    if TRAIN_WITH_PREPRO:
                        torch.save(Net.state_dict(), "supervise/drive_models/preprocessed-" + str(batch_N + 1) + ".torch")
                    else:
                        torch.save(Net.state_dict(), "supervise/drive_models/" + str(batch_N + 1) + ".torch")
            
            bar()