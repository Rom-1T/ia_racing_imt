import numpy as np
import cv2

class Preprocessing():
    
    def __init__(self, cfg):
        self.cropped_from_top = cfg.PREPROCESSED_CROP_FROM_TOP
        self.prepro = cfg.PREPROCESSING_METHOD

    def bnw(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def cropY(self, img, px_from_top):
        return img[px_from_top:np.shape(img)[0], :, :]

    def gaussian(self, img, ks=(3,3), sigmaX = 0):
        return cv2.GaussianBlur(img, ksize=ks, sigmaX=sigmaX)

    def threshold(self, img, thr = 195):
        return cv2.threshold(img, thr, 255, type=cv2.THRESH_BINARY)

    def canny(self, img):
        return cv2.Canny(img,100,200)
    
    def throttle150(self, image):
        img = image.copy()
        img = self.gaussian(img)
        img = self.bnw(img)
        img = self.threshold(img, 150)[1]
        return img

    def throttle195(self, image):
        img = image.copy()
        img = self.gaussian(img)
        img = self.bnw(img)
        img = self.threshold(img, 195)[1]
        return img

    def canny150(self, image):
        img = self.throttle150(image)
        img = self.canny(img)
        return img

    def canny195(self, image):
        img = self.throttle195(image)
        img = self.canny(img)
        return img

    def contour150(self, image):
        image_copy = image.copy()
        img = self.canny150(image)
        contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        return image_copy

    def contour195(self, image):
        image_copy = image.copy()
        img = self.canny195(image)
        contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        return image_copy

    def lines(self, image):
        img = image.copy()
        img = self.gaussian(img)
        img = self.bnw(img)
        edges = cv2.Canny(img, 150, 200, apertureSize = 3)
        minLineLength = 20
        maxLineGap = 5
        lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
        
        img = np.zeros((img.shape[0], img.shape[1], 3), dtype = "uint8")
        if not(lines is None):
            for x in range(0, len(lines)):
                for x1,y1,x2,y2 in lines[x]:
                    pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
                    cv2.polylines(img, [pts], True, (255, 0,0), 3)
            img[54:, 33:128, :] = 0 # Masque pour le parchoc
        return img


    def run(self, image_arr):
        img_preprocessed = image_arr[:]
        img_preprocessed = self.cropY(img_preprocessed, self.cropped_from_top)
        img_cropped = img_preprocessed.copy()
        
        if self.prepro == "throttle150":
            img_preprocessed = self.throttle150(img_preprocessed)
        elif self.prepro == "throttle195":
            img_preprocessed = self.throttle195(img_preprocessed)
        elif self.prepro == "canny150":
            img_preprocessed = self.canny150(img_preprocessed)
        elif self.prepro == "canny195":
            img_preprocessed = self.canny195(img_preprocessed)
        elif self.prepro == "contour150":
            img_preprocessed = self.contour150(img_preprocessed)
        elif self.prepro == "contour195":
            img_preprocessed = self.contour195(img_preprocessed)
        elif self.prepro == "lines":
            img_preprocessed = self.lines(img_preprocessed)

        return img_cropped, img_preprocessed
    