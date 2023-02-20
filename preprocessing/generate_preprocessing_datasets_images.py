import cv2
import numpy as np
import os
import preprocessing

def bnw(img, img_name, target_dir):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(target_dir + img_name, img)

def bnw_cropped(img, img_name, target_dir):
    img = preprocessing.cropY(img, 40)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(target_dir + img_name, img)

def cropped(img, img_name, target_dir):
    img = preprocessing.cropY(img, 40)
    cv2.imwrite(target_dir + img_name, img)

def cropped_bnw_threshold150(img, img_name, target_dir):
    img = preprocessing.cropY(img, 40)
    img = preprocessing.gaussian_filter(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = preprocessing.threshold_filter(img, 150)[1]
    cv2.imwrite(target_dir + img_name, img)

def cropped_bnw_threshold195(img, img_name, target_dir):
    img = preprocessing.cropY(img, 40)
    img = preprocessing.gaussian_filter(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = preprocessing.threshold_filter(img)[1]
    cv2.imwrite(target_dir + img_name, img)

def cropped_bnw_threshold150_canny(img, img_name, target_dir):
    img = preprocessing.cropY(img, 40)
    img = preprocessing.gaussian_filter(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = preprocessing.threshold_filter(img, 150)[1]
    img = preprocessing.canny(img)
    cv2.imwrite(target_dir + img_name, img)

def cropped_bnw_threshold195_canny(img, img_name, target_dir):
    img = preprocessing.cropY(img, 40)
    img = preprocessing.gaussian_filter(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = preprocessing.threshold_filter(img)[1]
    img = preprocessing.canny(img)
    cv2.imwrite(target_dir + img_name, img)

def cropped_canny150_contour(img, img_name, target_dir):
    img = preprocessing.cropY(img, 40)
    image_copy = img.copy()
    img = preprocessing.gaussian_filter(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = preprocessing.threshold_filter(img, 150)[1]
    img = preprocessing.canny(img)
    contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(target_dir + img_name, image_copy)

def cropped_canny195_contour(img, img_name, target_dir):
    img = preprocessing.cropY(img, 40)
    image_copy = img.copy()
    img = preprocessing.gaussian_filter(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = preprocessing.threshold_filter(img)[1]
    img = preprocessing.canny(img)
    contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(target_dir + img_name, image_copy)

def cropped_bnw_gaussian_canny_polylines(img, img_name, target_dir):
    img = cv2.imread(filename)
    img = preprocessing.cropY(img, 40)
    img = preprocessing.gaussian_filter(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,150,200,apertureSize = 3)
    minLineLength = 20
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
    
    img = np.zeros((img.shape[0], img.shape[1], 3), dtype = "uint8")
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
            pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
            cv2.polylines(img, [pts], True, (0,0,255), 3)
    img[54:, 33:128, :] = 0 # Masque pour le parchoc
    cv2.imwrite(target_dir + img_name, img)


SOURCE_DIR = "/Users/IMT_Atlantique/project_ia/data_remastered/raw/"
TARGET_DIR = "/Users/IMT_Atlantique/project_ia/data_remastered/"

for i in range(9483, 16697):
    filename = SOURCE_DIR + str(i) + "_cam_image_array_.jpg"
    if os.path.isfile(filename):
        img = cv2.imread(filename)
        # bnw(img, str(i)+"_cam_image_array_.jpg", TARGET_DIR + "/bnw/")
        # bnw_cropped(img, str(i)+"_cam_image_array_.jpg", TARGET_DIR + "/bnw_cropped/")
        # cropped(img, str(i)+"_cam_image_array_.jpg", TARGET_DIR + "/cropped/")
        # cropped_bnw_threshold150(img, str(i)+"_cam_image_array_.jpg", TARGET_DIR + "/cropped_bnw_threshold150/")
        # cropped_bnw_threshold195(img, str(i)+"_cam_image_array_.jpg", TARGET_DIR + "/cropped_bnw_threshold195/")
        # cropped_bnw_threshold195_canny(img, str(i)+"_cam_image_array_.jpg", TARGET_DIR + "/cropped_bnw_threshold195_canny/")
        # cropped_bnw_threshold150_canny(img, str(i)+"_cam_image_array_.jpg", TARGET_DIR + "/cropped_bnw_threshold150_canny/")
        # cropped_canny195_contour(img, str(i)+"_cam_image_array_.jpg", TARGET_DIR + "/cropped_canny195_contour/")
        cropped_canny150_contour(img, str(i)+"_cam_image_array_.jpg", TARGET_DIR + "/cropped_canny150_contour/")
        # cropped_bnw_gaussian_canny_polylines(img, str(i)+"_cam_image_array_.jpg", TARGET_DIR + "/cropped_bnw_gaussian_canny_polylines/")