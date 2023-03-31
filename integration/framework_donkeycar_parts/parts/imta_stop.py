__author__ = "Amaury COLIN"
__credits__ = "Amaury COLIN"
__date__ = "2023.03.17"
__version__ = "1.0.1"

import cv2
import numpy as np

class StopDetection():
    def __init__(self, cfg):
        
    # Specifying upper and lower ranges of color to detect in hsv format

        self.lowerYellow = np.array(cfg.STOP_LOWER_YELLOW)
        self.upperYellow = np.array(cfg.STOP_UPPER_YELLOW)
        self.area_threshold = cfg.STOP_YELLOW_AREA_MIN
        self.stop = False
        self.number_previous_images = cfg.STOP_DETECTION_PREVIOUS_IMG_BASE + 1
        self.previous_images = [None for _ in range(self.number_previous_images)]
        self.lap_counter = 0
        self.lap_counter_max = cfg.LAP_COUNTER_MAX
        self.crop_from_top = cfg.PREPROCESSED_CROP_FROM_TOP if cfg.PREPROCESSING else 0
        self.end = False
    
    # Inputs : image, throttle
    # Outputs : image, line_found, lap_counter, end
    
    def run(self, img_arr, img_arr_cropped, stopInactive):
        if not(stopInactive) and not(self.end):
            img_copy = img_arr.copy()
            if not(img_arr_cropped is None):
                img_cropped_copy = img_arr_cropped.copy()
            else : 
                img_cropped_copy = img_copy
                
            img = cv2.cvtColor(img_cropped_copy, cv2.COLOR_RGB2HSV) # Converting BGR image to HSV format
            img = cv2.blur(img, (5,5))  # Blur
            
            mask = cv2.inRange(img, self.lowerYellow, self.upperYellow) # Masking the image to find our color
            mask = cv2.erode(mask, None, iterations=4)
            mask = cv2.dilate(mask, None, iterations=4)
            image2 = cv2.bitwise_and(img, img, mask=mask)

            mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image
            
            line_found = 0
            # Finding position of all contours
            if len(mask_contours) != 0:
                for mask_contour in mask_contours:
                    if cv2.contourArea(mask_contour) > self.area_threshold:
                        line_found = 1
                        x, y, w, h = cv2.boundingRect(mask_contour)
                        cv2.rectangle(img_copy, (x, y + self.crop_from_top), (x + w, y + self.crop_from_top + h), (0, 0, 255), 3) #drawing rectangle
                        # print("Detection jaune avec une aire de :", cv2.contourArea(mask_contour))
            
            if line_found == 0:
                if all(self.previous_images):
                    print('*****************************************************')
                    print(f'*************** On commence le tour {self.lap_counter} ***************')
                    print('*****************************************************')
                    self.lap_counter += 1
            
            self.previous_images = self.previous_images[1: self.number_previous_images - 1] + [line_found]
            #print(self.previous_images, all(self.previous_images), line_found)
            
            if self.lap_counter > self.lap_counter_max:
                self.end = True
                if self.end:
                    print('*****************************************************')
                    print(f'************** La course est terminée **************')
                    print('*****************************************************')
                return img_copy,line_found, self.lap_counter, self.end
            
            return img_copy, line_found, self.lap_counter, self.end
        else:
            return img_arr, None, self.lap_counter, False