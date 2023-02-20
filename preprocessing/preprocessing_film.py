import cv2
import numpy as np
import os
import preprocessing

img_array = []
for i in range(9483, 16697):
    filename = "/Users/IMT_Atlantique/project_ia/data_race_images/images/"+str(i)+"_cam_image_array_.jpg"
    if os.path.isfile(filename):
        img = cv2.imread(filename)
        img = preprocessing.cropY(img, 40)
        image_copy = img.copy()
        img = preprocessing.gaussian_filter(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = preprocessing.threshold_filter(img, 150)[1]
        # img = preprocessing.canny(img)
        
        # contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        
        edges = cv2.Canny(img,150,200,apertureSize = 3)
        minLineLength = 20
        maxLineGap = 5
        lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
        
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # img = image_copy
        img = np.zeros((img.shape[0], img.shape[1], 3), dtype = "uint8")
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
                pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
                cv2.polylines(img, [pts], True, (0,0,255), 3)
        
        # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        img[54:, 33:128, :] = 0 # Masque pour le parchoc
        
        if len(img.shape) == 2:
            height, width = img.shape
            layers = 0
        else:
            height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


out = cv2.VideoWriter('preprocessing/videos/cropped_BnW_gaussian_canny_polyline.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 20, size, layers)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

# Cropped: 
# Crop : 40