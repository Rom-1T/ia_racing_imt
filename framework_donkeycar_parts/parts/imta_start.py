__author__ = "Wyatt MARIN et Malick BA, mise en forme Amaury COLIN"
__credits__ = ["Wyatt MARIN, Amaury COLIN", "Malick BA"]
__date__ = "2023.03.17"
__version__ = "1.0.1"
import cv2
import numpy as np

class IMTA_Start():
    
    def __init__(self, cfg):
        
    # Specifying upper and lower ranges of color to detect in hsv format

        self.lowerGreen = np.array(cfg.START_LOWER_GREEN)
        self.upperGreen = np.array(cfg.START_UPPER_GREEN)
        self.area_threshold = cfg.START_GREEN_AREA_MIN
        self.start = False
    
    def run(self, img_arr):
        img_copy = img_arr.copy()
        img = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV) # Converting BGR image to HSV format
        img = cv2.blur(img, (5,5))  # Blur
        
        mask = cv2.inRange(img, self.lowerGreen, self.upperGreen) # Masking the image to find our color
        mask = cv2.erode(mask, None, iterations=4)
        mask = cv2.dilate(mask, None, iterations=4)
        image2 = cv2.bitwise_and(img, img, mask=mask)

        mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image
        
        # Finding position of all contours
        if len(mask_contours) != 0:
            for mask_contour in mask_contours:
                if cv2.contourArea(mask_contour) > self.area_threshold:
                    x, y, w, h = cv2.boundingRect(mask_contour)
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle
                    if not(self.start):
                        print("Detection vert avec une aire de :", cv2.contourArea(mask_contour))
                        print('*****************************************************')
                        print('************** Demarrage de la voiture **************')
                        print('*****************************************************')
                        self.start = True
                        return img_copy, True
        return img_copy, False