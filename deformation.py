import cv2
import numpy as np
import os
from random import randint

from otsu_tresh import *

image_title = "dataset_mix_reel/data_set_ori_simu/10.jpg"
directory = 'dataset_mix_reel/data_set_ori_simu'
saving_directory="dataset_mix_reel/filtre_degrade/simu/"
saving_directory_test = "dataset_mix_reel/test/simu/"

def motion_blur_effect(img,kernel_size=None) :
    if kernel_size == None :
        kernel_size = randint(1,8)
    # Specify the kernel size.
    # The greater the size, the more the motion.

    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(img, -1, kernel_v)

    # Apply the horizontal kernel.
    horizonal_mb = cv2.filter2D(vertical_mb, -1, kernel_h)

    return horizonal_mb

i=0
j=0
for filename in os.scandir(directory):
    if j>50 :
        break
    if filename.is_file():
        if i== 5 :
            print(i)
            original = cv2.imread(filename.path, 0)
            # filtered = final_filter(original)
            motion_blur = motion_blur_effect(original)
            filtered = final_filter(motion_blur)
            i=0
            # cv2.imwrite(saving_directory_test+ "/original/"+filename.path.split("/")[-1],original)
            # cv2.imwrite(saving_directory_test + "/motion_blur/" + filename.path.split("/")[-1], motion_blur)
            # cv2.imwrite(saving_directory_test + "/filtered/" + filename.path.split("/")[-1], filtered)
        original = cv2.imread(filename.path, 0)
        # filtered = final_filter(original)
        motion_blur = motion_blur_effect(original)
        filtered = final_filter(motion_blur)
        # cv2.imwrite(saving_directory + "/original/" + filename.path.split("/")[-1], original)
        # cv2.imwrite(saving_directory + "/motion_blur/" + filename.path.split("/")[-1], motion_blur)
        # cv2.imwrite(saving_directory + "/filtered/" + filename.path.split("/")[-1], filtered)
        cv2.imshow("original",original)
        cv2.imshow("blur", motion_blur)
        cv2.imshow("filtered", filtered)
        cv2.waitKey(0)
        i+=1
        j+=1

