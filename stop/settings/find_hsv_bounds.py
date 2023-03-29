__author__ = "Amaury COLIN"
__credits__ = "Amaury COLIN"
__date__ = "2023.03.21"
__version__ = "1.0.1"

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk

win = tk.Tk()
win.geometry("700x750")

imgs = [
        cv2.imread(os.path.join(os.getcwd(), "../cars/imtaracing.local-mycar/data/tub_215_23-03-16/images/133_cam_image_array_.jpg")),
        cv2.imread(os.path.join(os.getcwd(), "../cars/imtaracing.local-mycar/data/tub_215_23-03-16/images/134_cam_image_array_.jpg")),
        cv2.imread(os.path.join(os.getcwd(), "../cars/imtaracing.local-mycar/data/tub_215_23-03-16/images/135_cam_image_array_.jpg")),
        cv2.imread(os.path.join(os.getcwd(), "../cars/imtaracing.local-mycar/data/tub_215_23-03-16/images/136_cam_image_array_.jpg")),
        cv2.imread(os.path.join(os.getcwd(), "../cars/imtaracing.local-mycar/data/tub_215_23-03-16/images/236_cam_image_array_.jpg")),
        cv2.imread(os.path.join(os.getcwd(), "../cars/imtaracing.local-mycar/data/tub_215_23-03-16/images/456_cam_image_array_.jpg")),
        cv2.imread(os.path.join(os.getcwd(), "../cars/imtaracing.local-mycar/data/tub_215_23-03-16/images/137_cam_image_array_.jpg")),
        cv2.imread(os.path.join(os.getcwd(), "../cars/imtaracing.local-mycar/data/tub_215_23-03-16/images/138_cam_image_array_.jpg"))
    ]
imgs_hsv = [None for _ in range(len(imgs))]
im = [None for _ in range(len(imgs))]
imgtk = [None for _ in range(len(imgs))]
imgLabel = [None for _ in range(len(imgs))]

for i in range(len(imgs)):
    imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
    imgs_hsv[i] = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2HSV)

    im[i] = Image.fromarray(imgs[i])
    imgtk[i] = ImageTk.PhotoImage(image=im[i])

    r, c = divmod(i, 4)
    imgLabel[i] = tk.Label(win, image= imgtk[i])
    imgLabel[i].grid(row = r, column = c)

def show_values(x):
    global imgs, imgs_hsv
    # print(lower_H.get(), lower_S.get(), lower_V.get(), upper_H.get(), upper_S.get(), upper_V.get())
    
    lower = np.array([lower_H.get(), lower_S.get(), lower_V.get()])
    upper = np.array([upper_H.get(), upper_S.get(), upper_V.get()])
    
    mask = [None for _ in range(len(imgs))]
    img_copy = [None for _ in range(len(imgs))]
    
    for i in range(len(imgs)):
        mask = cv2.inRange(imgs_hsv[i], lower, upper)
        mask = cv2.erode(mask, None, iterations=4)
        mask = cv2.dilate(mask, None, iterations=4)
        image2 = cv2.bitwise_and(imgs_hsv[i], imgs_hsv[i], mask=mask)

        mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image
        img_copy = imgs[i].copy()
        
        if len(mask_contours) != 0:
            for mask_contour in mask_contours:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 3)

        im = Image.fromarray(img_copy)
        imgtk[i] = ImageTk.PhotoImage(image=im)
        imgLabel[i].configure(image=imgtk[i])
        imgLabel[i].image = imgtk[i]

lower_H = tk.Scale(win, from_=0, to=100, orient=tk.HORIZONTAL, command=show_values, length=600, label="H borne inf")
lower_H.grid(row = divmod(len(imgs), 4)[0] + 1, columnspan=4)
lower_S = tk.Scale(win, from_=0, to=255, orient=tk.HORIZONTAL, command=show_values, length=600, label="S borne inf")
lower_S.grid(row = divmod(len(imgs), 4)[0] + 2, columnspan=4)
lower_V = tk.Scale(win, from_=0, to=255, orient=tk.HORIZONTAL, command=show_values, length=600, label="V borne inf")
lower_V.grid(row = divmod(len(imgs), 4)[0] + 3, columnspan=4)

upper_H = tk.Scale(win, from_=0, to=100, orient=tk.HORIZONTAL, command=show_values, length=600, label="H borne sup")
upper_H.set(100)
upper_H.grid(row = divmod(len(imgs), 4)[0] + 4, columnspan=4)
upper_S = tk.Scale(win, from_=0, to=255, orient=tk.HORIZONTAL, command=show_values, length=600, label="S borne sup")
upper_S.set(255)
upper_S.grid(row = divmod(len(imgs), 4)[0] + 5, columnspan=4)
upper_V = tk.Scale(win, from_=0, to=255, orient=tk.HORIZONTAL, command=show_values, length=600, label="V borne sup")
upper_V.set(255)
upper_V.grid(row = divmod(len(imgs), 4)[0] + 6, columnspan=4)

show_values(None)

win.mainloop()