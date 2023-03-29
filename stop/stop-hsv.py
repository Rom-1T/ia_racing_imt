__author__ = "Amaury COLIN"
__credits__ = "Amaury COLIN"
__date__ = "2023.03.14"
__version__ = "1.0.1"

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Specifying upper and lower ranges of color to detect in hsv format
lower = np.array([30, 53, 74])
upper = np.array([36, 255, 255]) # (These ranges will detect Yellow)

IMG_DIR_SANS = "./stop/datasets/images_reelles/dataset_sigma/train/class_0/"
IMG_DIR_AVEC = "./stop/datasets/images_reelles/dataset_sigma/train/class_1/"
IMWRITE_DIR = "./stop/results/img_results_hsv/"


imgs = os.scandir(IMG_DIR_SANS)

c = 0
tot = 0

for image in imgs:
    if "jpg" in image.name:
        tot += 1
        video = cv2.imread(os.path.join(IMG_DIR_SANS, image.name))

        video = video[40:np.shape(video)[0], :, :]
        img = cv2.cvtColor(video, cv2.COLOR_BGR2HSV) # Converting BGR image to HSV format
        img=cv2.blur(img, (5,5))  # Blur

        mask = cv2.inRange(img, lower, upper) # Masking the image to find our color

        mask = cv2.erode(mask, None, iterations=4)
        mask = cv2.dilate(mask, None, iterations=4)
        image2 = cv2.bitwise_and(video, video, mask=mask)

        mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image
        
        # Finding position of all contours
        if len(mask_contours) != 0:
            for mask_contour in mask_contours:
                if cv2.contourArea(mask_contour) > 600:
                    x, y, w, h = cv2.boundingRect(mask_contour)
                    cv2.rectangle(video, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle
                    c+= 1
                    cv2.imwrite(IMWRITE_DIR + "class_0/" + image.name, cv2.cvtColor(video, cv2.COLOR_BGR2RGB))
                    

        cv2.imshow("mask image", mask) # Displaying mask image

        cv2.imshow("window", video) # Displaying webcam image

        cv2.waitKey(1)

print(f'Nombre de détection de ligne pour ABSENCE : {c}/{tot}')


imgs = os.scandir(IMG_DIR_AVEC)

c = 0
tot = 0

for image in imgs:
    if "jpg" in image.name:
        tot += 1
        video = cv2.imread(os.path.join(IMG_DIR_AVEC, image.name))

        video = video[40:np.shape(video)[0], :, :]
        img = cv2.cvtColor(video, cv2.COLOR_BGR2HSV) # Converting BGR image to HSV format
        img=cv2.blur(img, (5,5))  # Blur

        mask = cv2.inRange(img, lower, upper) # Masking the image to find our color

        mask = cv2.erode(mask, None, iterations=4)
        mask = cv2.dilate(mask, None, iterations=4)
        image2 = cv2.bitwise_and(video, video, mask=mask)

        mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image
        
        # Finding position of all contours
        if len(mask_contours) != 0:
            for mask_contour in mask_contours:
                if cv2.contourArea(mask_contour) > 600:
                    x, y, w, h = cv2.boundingRect(mask_contour)
                    cv2.rectangle(video, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle
                    c+= 1
                    cv2.imwrite(IMWRITE_DIR + "class_1/" + image.name, cv2.cvtColor(video, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(IMWRITE_DIR + "class_1/ignore/" + image.name, cv2.cvtColor(video, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(IMWRITE_DIR + "class_1/ignore/" + image.name, cv2.cvtColor(video, cv2.COLOR_RGB2BGR))
                    

        cv2.imshow("mask image", mask) # Displaying mask image

        cv2.imshow("window", video) # Displaying webcam image

        cv2.waitKey(1)

print(f'Nombre de détection de ligne pour PRÉSENCE : {c}/{tot}')

print(cv2.cvtColor(np.array([[[204, 204, 84]]], dtype=np.uint8), cv2.COLOR_RGB2HSV))
print(cv2.cvtColor(np.array([[[205, 219, 112]]], dtype=np.uint8), cv2.COLOR_RGB2HSV))
print(cv2.cvtColor(np.array([[[277, 176, 86]]], dtype=np.uint8), cv2.COLOR_RGB2HSV))