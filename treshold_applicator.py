import numpy as np
import matplotlib.pyplot as plt
import cv2

def crop_img(img=None, crop_pourcent=75, keep="down", suppress_shadow=True):
    ratio = crop_pourcent / 100
    nb_line_keep = int(len(img) * ratio)
    if keep == "down":
        if suppress_shadow:
            new_img = img[(int(len(img) - nb_line_keep)):int(len(img) * 0.9)]
        else:
            new_img = img[(int(len(img) - nb_line_keep)):]
    else:
        new_img = img[:nb_line_keep]
    return new_img

def treshold_applicator(img,min_value,max_value) :
    img_copy = np.copy(img)
    for i in range (0,len(img)):
        for j in range (0,len(img[i])) :
            if np.mean(img[i][j]) < min_value :
                pixel = [0,0,0]
                img_copy[i][j] = pixel
            elif np.mean(img[i][j]) >= max_value :
                pixel = [255,255,255]
                img_copy[i][j] = pixel
    return img_copy

def simple_treshold_applicator(img,value) :
    img_copy = np.copy(img)
    for i in range (0,len(img)):
        for j in range (0,len(img[i])) :
            if np.mean(img[i][j]) <= value :
                pixel = [0,0,0]
                img_copy[i][j] = pixel
            else:
                pixel = [255,255,255]
                img_copy[i][j] = pixel
    return img_copy

def test_treshold() :
    # create figure
    fig = plt.figure(figsize=(11, 8))
    # setting values to rows and column variables
    rows = 3
    columns = 3

    cam_img = cv2.imread("data/test.jpg", cv2.IMREAD_COLOR)
    processed_img1 = treshold_applicator(cam_img,50,100)
    processed_img2 = treshold_applicator(cam_img,50,150)
    processed_img3 = treshold_applicator(cam_img,50,200)
    processed_img4 = treshold_applicator(cam_img,100,200)
    processed_img5 = simple_treshold_applicator(cam_img,150)
    processed_img6 = simple_treshold_applicator(cam_img,170)
    processed_img7 = simple_treshold_applicator(cam_img,125)
    fig.add_subplot(rows, columns, 1)
    plt.imshow(cam_img, cmap='gray')
    plt.title("Original")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(processed_img1, cmap='gray')
    plt.title("min_value=50 max_value=100")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(processed_img2, cmap='gray')
    plt.title("min_value=50 max_value=150")

    fig.add_subplot(rows, columns, 4)
    plt.imshow(processed_img3, cmap='gray')
    plt.title("min_value=50 max_value=200")

    fig.add_subplot(rows, columns, 5)
    plt.imshow(processed_img4, cmap='gray')
    plt.title("min_value=100 max_value=200")

    fig.add_subplot(rows, columns, 6)
    plt.imshow(processed_img5, cmap='gray')
    plt.title("min_value=max_value=150")

    fig.add_subplot(rows, columns, 7)
    plt.imshow(processed_img6, cmap='gray')
    plt.title("min_value=max_value=170")

    fig.add_subplot(rows, columns, 8)
    plt.imshow(processed_img7, cmap='gray')
    plt.title("min_value=max_value=125")

    plt.show()

    print(cam_img[0][0])
    print(processed_img1[0][0])

def test_fisheye():
    # create figure

    K = np.array([[689.21, 0., 1295.56],
                  [0., 690.48, 942.17],
                  [0., 0., 1.]])

    # zero distortion coefficients work well for this image
    D = np.array([0.1, 0.1, 0.1, 0.1])

    # use Knew to scale the output
    Knew = K.copy()
    Knew[(0, 1), (0, 1)] = 0.4 * Knew[(0, 1), (0, 1)]


    img = cv2.imread("data/fisheye.jpg")
    DIM = np.size(img)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM,cv2.CV_16SC2)
    img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite('data/fisheye_sample_undistorted.jpg', img_undistorted)
    plt.imshow(img_undistorted, cmap='gray')
    plt.show()
    #cv2.imshow('undistorted', img_undistorted)
