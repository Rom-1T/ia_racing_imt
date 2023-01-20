import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

IMAGE = cv2.imread('/Users/IMT_Atlantique/project_ia/cars/mysim/data/images/119_cam_image_array_.jpg')
IMAGE_BNW = cv2.imread('/Users/IMT_Atlantique/project_ia/cars/mysim/data/images/119_cam_image_array_.jpg', cv2.IMREAD_GRAYSCALE)

AFFICHAGE = True
COLS = 3

def cropY(img, px_from_top):
    if len(np.shape(img)) == 3:
        return img[px_from_top:img.shape[0], :, :]
    return img[px_from_top:img.shape[0], :]

def gaussian_filter(img, ksize=(3,3), sigmaX = 0):
    return cv2.GaussianBlur(img, ksize=ksize, sigmaX=sigmaX)

def threshold_filter(img):
    return cv2.threshold(img, 195, 255, type=cv2.THRESH_BINARY)

def canny(img):
    return cv2.Canny(img,100,200)

images = {
    'Original' : IMAGE,
}

images['IMAGE_BNW'] = gaussian_filter(IMAGE_BNW)
images['Cropped'] = cropY(IMAGE, 40)
images['Cropped BW'] = cropY(images['IMAGE_BNW'], 45)
images['Cropped Gaussien'] = gaussian_filter(images['Cropped'])
images['Threshold'] = threshold_filter(cv2.cvtColor(images['Cropped Gaussien'], cv2.IMREAD_GRAYSCALE))[1]
images['Threshold BW'] = threshold_filter(images['Cropped BW'])[1]
images['Canny'] = canny(images['Cropped Gaussien'])

transfos = images.keys()



if AFFICHAGE:
    ROWS = int(len(images)/COLS) + 1
    fig = plt.figure()
    c = 0
    for img in transfos:
        c += 1
        fig.add_subplot(ROWS, COLS, c)
        try:
            plt.imshow(cv2.cvtColor(images[img], cv2.COLOR_BGR2RGB))
        except:
            pass
        try: 
            plt.imshow(images[img], cmap='gray')
        except:
            pass
        plt.axis("off")
        plt.title(img)

    plt.show()

