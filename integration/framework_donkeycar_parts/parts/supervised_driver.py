# Load and run neural network and make preidction
import json
import math
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import cv2


class SupervisedDrive():
    
    def __init__(self, cfg):
        self.device = cfg.SUPERVISED_DEVICE
        
        self.model = torchvision.models.resnet18(pretrained=True) # Load net
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=2, bias=True) # Set final layer to predict one value
        self.model = self.model.to(self.device) # Assign net to gpu or cpu

        self.model.load_state_dict(torch.load(cfg.SUPERVISED_STATE_DICT_PATH))

        self.throttle_max = cfg.SUPERVISED_THROTTLE_MAX
        self.angle_max = cfg.SUPERVISED_ANGLE_MAX
        
        self.previous_throttle = 0
        self.previous_angle = 0

    def image_to_tensor(self, cam_image_arr):
        print("img shape", np.shape(cam_image_arr))
        transformImg=tf.Compose([ tf.ToPILImage(),
                             tf.ToTensor()]) 
        camera_image_tensor = transformImg(cam_image_arr) # Transform to pytorch
        print("tensor shape", np.shape(camera_image_tensor))
        camera_image_tensor = torch.unsqueeze(camera_image_tensor, 0)
        return camera_image_tensor


    def run(self, cam_image_arr):
        
        print("img shape run", np.shape(cam_image_arr))
        image_tensor = self.image_to_tensor(cam_image_arr)
        print("tensor shape unsqueez", np.shape(image_tensor))
        image_tensor = torch.autograd.Variable(image_tensor, requires_grad=False).to(self.device) # Load image
        
        outputs = self.model(image_tensor)
        
        throttle, angle = float(outputs[0][0]), float(outputs[0][1])
        if throttle > 0:
            throttle = min(self.throttle_max, throttle)
        else:
            throttle = -1 * min(self.throttle_max, -1 * throttle)
        
        if angle > 0:
            angle = min(self.angle_max, angle)
        else:
            angle = -1 * min(self.angle_max, -1 * angle)
        print(throttle, self.previous_throttle, angle, self.previous_angle)
        
        throttle_decision = throttle
        angle_decision = angle
        
        self.previous_throttle = throttle
        self.previous_angle = angle
        return throttle_decision, angle_decision