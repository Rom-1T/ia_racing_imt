import cv2
from random import randint
import numpy as np
import albumentations as A
def motion_blur_effect(img,kernel_size=None) :
    if kernel_size == None :
        kernel_size = randint(1,8)
    # Specify the kernel size.
    # The greater the size, the more the motion.

    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(img, -1, kernel_v)

    # Apply the horizontal kernel.
    horizonal_mb = cv2.filter2D(vertical_mb, -1, kernel_h)

    return horizonal_mb
def final_filter(image) :
    image = cv2.GaussianBlur(image, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    filter = cv2.addWeighted(threshold,1,image_result,0.5,0.0)
    return(filter)

def degradation(img):
    x = np.random.randint(50, 100)
    img = A.RandomSunFlare(flare_roi=(0, 0, 1, 1), num_flare_circles_lower=1, num_flare_circles_upper=3,
                           src_color=(255, 255, 255), src_radius=x, always_apply=False)(image=img)['image']
    y = np.random.randint(0, 5)
    t1 = A.RandomSunFlare(flare_roi=(0, 0, 1, 1), num_flare_circles_lower=1, num_flare_circles_upper=3,
                          src_color=(255, 255, 255), src_radius=x, always_apply=True)
    t2 = A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1,
                      drop_color=(255, 255, 255), blur_value=1, brightness_coefficient=1, rain_type=None,
                      always_apply=True)
    t3 = A.CoarseDropout(max_holes=10, max_height=10, max_width=15, min_holes=1, min_height=1, min_width=1,
                         fill_value=(255, 255, 255), mask_fill_value=None, always_apply=True)
    t4_ = A.MotionBlur(blur_limit=(7, 7), always_apply=True)
    t4 = A.Compose([t4_, t4_])
    transfos = A.SomeOf([t1, t2, t3, t4], n=y)
    img = transfos(image=img)['image']
    return img

