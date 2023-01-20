from gc import callbacks
import cv2

class Preprocessing():
    
    def __init__(self, cfg):
        self.cropped_from_top = cfg.PREPROCESSED_CROP_FROM_TOP
        self.transformations = cfg.PREPROCESSED_METHODS

    def cropY(self, img, px_from_top):
        return img[px_from_top:img.shape[0], :, :]

    def gaussian(self, img, ks=(3,3), sigmaX = 0):
        return cv2.GaussianBlur(img, ksize=ks, sigmaX=sigmaX)

    def threshold(self, img):
        return cv2.threshold(img, 195, 255, type=cv2.THRESH_BINARY)

    def canny(self, img):
        return cv2.Canny(img,100,200)

    def run(self, image_arr):
        img_preprocessed = image_arr[:]
        img_preprocessed = self.cropY(img_preprocessed, self.cropped_from_top)
        
        for prepro in self.transformations:
            if prepro == "gaussian":
                img_preprocessed = self.gaussian(img_preprocessed)
            elif prepro == "canny":
                img_preprocessed = self.canny(img_preprocessed)
    
        return img_preprocessed
    