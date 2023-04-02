# Load and run neural network and make preidction
import json
import math
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import cv2
import os
import matplotlib.pyplot as plt


class SupervisedDrive():
    
    def __init__(self, cfg):
        self.device = cfg['SUPERVISED_DEVICE']
        
        self.model = torchvision.models.resnet18(pretrained=True) # Load net
        self.model.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True) # Set final layer to predict one value
        self.model = self.model.to(self.device) # Assign net to gpu or cpu

        self.model.load_state_dict(torch.load(cfg['SUPERVISED_STATE_DICT_PATH']))

    def image_to_tensor(self, cam_image_arr):
        transformImg=tf.Compose([ tf.ToPILImage(),
                             tf.ToTensor()]) 
        camera_image_tensor = transformImg(cam_image_arr) # Transform to pytorch
        camera_image_tensor = torch.unsqueeze(camera_image_tensor, 0)
        
        return camera_image_tensor


    def run(self, cam_image_arr):
        
        image_tensor = self.image_to_tensor(cam_image_arr)
        
        image_tensor = torch.autograd.Variable(image_tensor, requires_grad=False).to(self.device) # Load image
        
        outputs = self.model(image_tensor)
        
        throttle, angle = float(outputs[0][0]), float(outputs[0][1])
        
        return throttle, angle
    

if __name__ == "__main__":
    cfg = {
        'SUPERVISED_DEVICE': "cpu",
        'SUPERVISED_STATE_DICT_PATH': os.getcwd() + '/supervise/drive_models/999.torch'
    }
    
    s_driver = SupervisedDrive(cfg)
    
    img = cv2.imread(os.getcwd() + "/supervise/dataset_drive/lap_001-59_cam-image_array_.jpg")
    # plt.imshow(img)
    # plt.show()
    
    print(s_driver.run(img))