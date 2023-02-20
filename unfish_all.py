# import required module
import os

# You should replace these 3 lines with the output in calibration step
import sys
import numpy as np
import cv2
import time
from treshold_applicator import treshold_applicator

# DIM=(1280, 720)
# K=np.array([[612.0192249166925, 0.0, 658.5452739196181], [0.0, 607.029492138849, 363.9068387400568], [0.0, 0.0, 1.0]])
# D=np.array([[0.00806244711831274], [-0.15761080004651892], [0.3619475431378882], [-0.32127145742471525]])

DIM=(160, 120)
K=np.array([[76.40210369377033, 0.0, 85.17741324657462], [0.0, 75.55575570884872, 61.5111216120113], [0.0, 0.0, 1.0]])
D=np.array([[0.032858036745614], [-0.09739958496116238], [0.07344214252074698], [-0.02977154953395648]])

def filter_img(img=None, v_min=100, v_max=200, filter_type="gaussian", nbr_img=0, random_nbr=0, alpha=0.4, beta=0.6,
               log_activated=False):
    pre_filter = cv2.GaussianBlur(img, (3, 3), 0)
    if filter_type == "gaussian":
        filter = pre_filter
    elif filter_type == "median":
        filter = cv2.medianBlur(img, 3)
    elif filter_type == "canny":
        edges_1D = cv2.Canny(pre_filter, v_min, v_max)
        edges_3D = np.stack((edges_1D, edges_1D, edges_1D), axis=2)
        filter = cv2.addWeighted(pre_filter, alpha, edges_3D, beta, 0)
    elif filter_type == "laplacian":
        edges = cv2.Laplacian(pre_filter, ddepth=cv2.CV_8U)
        filter = cv2.addWeighted(pre_filter, alpha, edges, beta, 0)
    elif filter_type == "sobel":
        filter = cv2.Sobel(pre_filter, ddepth=cv2.CV_64F, dx=1, dy=0)
    elif filter_type == "erode":
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(pre_filter, kernel)
        filter = cv2.dilate(eroded, kernel)
    elif filter_type == "erode+laplacian":
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(pre_filter, kernel)
        filter = cv2.dilate(eroded, kernel)
        filter = cv2.Laplacian(filter, ddepth=cv2.CV_64F)
    elif filter_type == "laplacian+erode":
        pre_filter = cv2.Laplacian(pre_filter, ddepth=cv2.CV_64F)
        kernel = np.ones((5, 5), np.uint8)
        filter = cv2.dilate(pre_filter, kernel)
    elif filter_type == "log":
        filter = img
    else:
        filter_type = "none"
        filter = pre_filter
    if log_activated:
        directory_path = f"log_img/log_{filter_type}_{random_nbr}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        img_to_save = f"/img_{nbr_img}.jpg"
        path = directory_path + img_to_save
        cv2.imwrite(path, filter)
    return filter


def undistort(img_path=None):
    img = cv2.imread(img_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    print("DONE:  ", img_path)
    return undistorted_img


# assign directory
directory = 'dataset_mix_reel/data_set_reel'
saving_dir = 'dataset_mix_reel/data_set_filtered/reel'
saving_dir_ori = 'dataset_mix_reel/data_set_ori_reel'

# iterate over files in
# that directory
i=0
for filename in os.scandir(directory):
    if filename.is_file():
        img = cv2.imread(filename.path, 0)
        # img = cv2.medianBlur(img, 5)
        undistor_img = undistort(filename.path)
        undistor_img = cv2.GaussianBlur(undistor_img,(5,5),0)
        filtered = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        #undistor_img = treshold_applicator(undistor_img,min_value=150,max_value=200)
        #filtered = filter_img(img=undistor_img,filter_type='')
        cv2.imwrite(saving_dir + f'/{i}.jpg',filtered)
        cv2.imwrite(saving_dir_ori + f'/{i}.jpg', img)
        i += 1
        print(filename.path)


