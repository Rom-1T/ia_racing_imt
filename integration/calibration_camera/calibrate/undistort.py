# You should replace these 3 lines with the output in calibration step
import sys
import numpy as np
import cv2
import time


# DIM=(1280, 720)
# K=np.array([[612.0192249166925, 0.0, 658.5452739196181], [0.0, 607.029492138849, 363.9068387400568], [0.0, 0.0, 1.0]])
# D=np.array([[0.00806244711831274], [-0.15761080004651892], [0.3619475431378882], [-0.32127145742471525]])

DIM=(160, 120)
K=np.array([[76.40210369377033, 0.0, 85.17741324657462], [0.0, 75.55575570884872, 61.5111216120113], [0.0, 0.0, 1.0]])
D=np.array([[0.032858036745614], [-0.09739958496116238], [0.07344214252074698], [-0.02977154953395648]])


def undistort(saving_path = "/unfish",img_path=None):
    img = cv2.imread(img_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(saving_path + img_path, undistorted_img)
    print("DONE:  ", img_path)


