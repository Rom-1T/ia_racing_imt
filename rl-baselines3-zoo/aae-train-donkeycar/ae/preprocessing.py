import cv2
from random import randint
import numpy as np
import albumentations as A

# ============ DonkeyCar Config ================== #
# Raw camera input

CAMERA_HEIGHT = 120
CAMERA_WIDTH = 160

MARGIN_TOP = CAMERA_HEIGHT // 3
# MARGIN_TOP = 0

# ============ End of DonkeyCar Config ============ #

# Camera max FPS
FPS = 40


# Region Of Interest
# r = [margin_left, margin_top, width, height]
ROI = [0, MARGIN_TOP, CAMERA_WIDTH, CAMERA_HEIGHT - MARGIN_TOP]

# Fixed input dimension for the autoencoder
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 80
N_CHANNELS = 3
RAW_IMAGE_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH, N_CHANNELS)
INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)

# Arrow keys, used by opencv when displaying a window

DIM = (160, 120)
K = np.array([[76.40210369377033, 0.0, 85.17741324657462], [0.0, 75.55575570884872, 61.5111216120113], [0.0, 0.0, 1.0]])
D = np.array([[0.032858036745614], [-0.09739958496116238], [0.07344214252074698], [-0.02977154953395648]])


def lines(img, edges):
    minLineLength = 20
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 30, minLineLength, maxLineGap)

    img = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")
    if not (lines is None):
        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                pts = np.array([[x1, y1], [x2, y2]], np.int32)
                cv2.polylines(img, [pts], True, (255, 0, 0), 3)
    return img


def undistort(img_array):
    # print("UNDISTORT")
    # img_array = np.array(img_array)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img_array, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def crop_img(image,roi) :
    r = ROI
    image = image[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
    im = image
    return im
def processing_line_v2(image, mode="edge", reel=True, epaisseur=1):
    image = crop_img(image, ROI)
    if reel:
        image = undistort(image)
    edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.GaussianBlur(edge_image, (3, 3), 1)
    edge_image = cv2.Canny(edge_image, 150, 200, apertureSize=3)
    edge_image = cv2.dilate(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=epaisseur
    )
    edge_image = cv2.erode(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=epaisseur
    )
    if mode == "edge":
        return edge_image
    else:
        return lines(image, edge_image)


def motion_blur_effect(img, kernel_size=None):
    if kernel_size == None:
        kernel_size = randint(1, 8)
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


def final_filter(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    filter = cv2.addWeighted(threshold, 1, image_result, 0.5, 0.0)
    return (filter)


def degradation(img):
    x = np.random.randint(10, 25)
    img = A.RandomSunFlare(flare_roi=(0, 0, 1, 1), num_flare_circles_lower=1, num_flare_circles_upper=3,
                           src_color=(255, 255, 255), src_radius=x, always_apply=False)(image=img)['image']
    y = np.random.randint(0, 2)
    t1 = A.RandomSunFlare(flare_roi=(0, 0, 1, 1), num_flare_circles_lower=1, num_flare_circles_upper=3,
                          src_color=(255, 255, 255), src_radius=x, always_apply=True)
    # t2 = A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1,drop_color=(255, 255, 255), blur_value=1, brightness_coefficient=1, rain_type=None,always_apply=True)
    t3 = A.CoarseDropout(max_holes=10, max_height=10, max_width=15, min_holes=1, min_height=1, min_width=1,
                         fill_value=(255, 255, 255), mask_fill_value=None, always_apply=True)
    t4_ = A.MotionBlur(blur_limit=(7, 7), always_apply=True)
    t4 = A.Compose([t4_, t4_])
    transfos = A.SomeOf([t1, t3, t4], n=y)
    img = transfos(image=img)['image']
    return img
