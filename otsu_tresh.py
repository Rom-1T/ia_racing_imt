import cv2
import numpy as np
import os

is_normalized = False

image_title = "dataset_mix_reel/data_set_ori_reel/0.jpg"
directory = 'dataset_mix_reel/data_set_ori_reel'
saving_directory="dataset_mix_reel/filtre/otsu/reel/"

def otsu_calc_hand(image_title) :
    # Read the image in a grayscale mode
    image = cv2.imread(image_title, 0)

    # Apply GaussianBlur to reduce image noise if it is required
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    print(threshold)
    #print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold
def otsu_calc(image_title):
    # Read the image in a grayscale mode
    image = cv2.imread(image_title, 0)

    # Apply GaussianBlur to reduce image noise if it is required
    image = cv2.GaussianBlur(image, (5, 5), 0)

    otsu_threshold, image_result = cv2.threshold(
        image, 0, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    print("Obtained threshold: ", otsu_threshold)
    return otsu_threshold,image_result

def gaussian_calc(image_title):
    # Read the image in a grayscale mode
    image = cv2.imread(image_title, 0)

    # Apply GaussianBlur to reduce image noise if it is required
    image = cv2.GaussianBlur(image, (5, 5), 0)

    threshold = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,5)

    #threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 8)

    dst = cv2.addWeighted(threshold,1,otsu_calc(image_title)[1],0.5,0.0)
    return dst


def dilatation(image_title) :
    src = cv2.imread(image_title,0)
    src = cv2.GaussianBlur(src, (5, 5), 0)
    LoG = cv2.Laplacian(src, cv2.CV_16S)
    minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3, 3)))
    maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((5, 5)))
    zeroCross = np.logical_or(np.logical_and(minLoG < 0, LoG > 0), np.logical_and(maxLoG > 0, LoG < 0))
    filter = cv2.morphologyEx(maxLoG, cv2.MORPH_CLOSE, np.ones((5, 5)))
    return filter


def mean_otsu(directory,saving_directory=None) :
    liste_otsu = []
    for filename in os.scandir(directory):
        if filename.is_file():
            otsu_threshold, image_result = otsu_calc(filename.path)
            liste_otsu.append(otsu_threshold)
            print(type(image_result))
            if saving_directory is not None :
                print(saving_directory + filename.path.split("/")[-1])
                cv2.imwrite(saving_directory + filename.path.split("/")[-1], image_result)
    print("Otsu's algorithm implementation thresholding result: ", np.mean(liste_otsu))

def mean_filter(directory,saving_directory) :
    for filename in os.scandir(directory):
        if filename.is_file():
            image_result = gaussian_calc(filename.path)
            print(type(image_result))
            if saving_directory is not None :
                print(saving_directory + filename.path.split("/")[-1])
                cv2.imwrite(saving_directory + filename.path.split("/")[-1], image_result)


def final_filter(image) :
    image = cv2.GaussianBlur(image, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    filter = cv2.addWeighted(threshold,1,image_result,0.5,0.0)
    return(filter)

