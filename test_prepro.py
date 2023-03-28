import cv2
import numpy as np

DIM = (160, 120)
K = np.array([[76.40210369377033, 0.0, 85.17741324657462], [0.0, 75.55575570884872, 61.5111216120113], [0.0, 0.0, 1.0]])
D = np.array([[0.032858036745614], [-0.09739958496116238], [0.07344214252074698], [-0.02977154953395648]])

def cropY(img, px_from_top):
    return img[px_from_top:np.shape(img)[0], :, :]

def undistort(img_array):
    # print("UNDISTORT")
    # img_array = np.array(img_array)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img_array, map1, map2, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def processing_line_v2(image, mode="edge", reel=True, epaisseur=1):
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


def final_filter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    filter = cv2.addWeighted(threshold, 1, image_result, 0.5, 0.0)
    # filter =  cv2.cvtColor(filter, cv2.COLOR_GRAY2BGR)
    return (filter)


def crop(image, ROI):
    # Crop
    # Region of interest
    r = ROI
    image = image[int(r[1]): int(r[1] + r[3]), int(r[0]): int(r[0] + r[2])]
    return image

raw_reel = cv2.imread("file/reel.jpg")
raw_simu = cv2.imread("file/simulateur.jpg")
saving_dic_reel = "examples/reel/"
saving_dic_simu = "examples/simu/"

# IMAGE REEL
undistort_reel = undistort(raw_reel)
cropped = cropY(undistort_reel,40)
cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
gaussian_blur_reel = cv2.GaussianBlur(cropped, (3, 3), 1)
gaussian_thresh_reel = cv2.adaptiveThreshold(cropped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 15, 5)
otsu_thresh_reel = cv2.threshold(cropped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )[1]
final_filter_reel = final_filter(cropped)
canny_reel = processing_line_v2(cropped, mode="edge", reel=False, epaisseur=2)
hough_reel = processing_line_v2(cropped, mode="lines", reel=False, epaisseur=2)
cv2.imshow("Original", raw_reel)
cv2.imshow("Cropped", cropped)
cv2.imshow("Gaussian blur", gaussian_blur_reel)
cv2.imshow("Gaussian Threshold", gaussian_thresh_reel)
cv2.imshow("Otsu Threshold", otsu_thresh_reel)
cv2.imshow("Final Filter", final_filter_reel)
cv2.imshow("Canny", canny_reel)
cv2.imshow("Hough", hough_reel)
cv2.waitKey(0)

#Save
cv2.imwrite(saving_dic_reel + "cropped.jpg", cropped)
cv2.imwrite(saving_dic_reel + "gaussian_blur_reel.jpg", gaussian_blur_reel)
cv2.imwrite(saving_dic_reel + "gaussian_thresh_reel.jpg", gaussian_thresh_reel)
cv2.imwrite(saving_dic_reel + "otsu_thresh_reel.jpg", otsu_thresh_reel)
cv2.imwrite(saving_dic_reel + "final_filter_reel.jpg", final_filter_reel)
cv2.imwrite(saving_dic_reel + "canny_reel.jpg", canny_reel)
cv2.imwrite(saving_dic_reel + "hough_reel.jpg", hough_reel)



# IMAGE SIMU
cropped_simu = cropY(raw_simu,40)
cropped_gray_simu = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
gaussian_blur_simu = cv2.GaussianBlur(cropped_simu, (3, 3), 1)
gaussian_thresh_simu = cv2.adaptiveThreshold(cropped_gray_simu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                             15, 5)
otsu_thresh_simu = cv2.threshold(cropped_gray_simu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )[1]
final_filter_simu = final_filter(cropped_simu)
canny_simu = processing_line_v2(cropped_simu, mode="edge", reel=False, epaisseur=2)
hough_simu = processing_line_v2(cropped_simu, mode="lines", reel=False, epaisseur=2)

cv2.imshow("Original", raw_simu)
cv2.imshow("Cropped", cropped_simu)
cv2.imshow("Gaussian blur", gaussian_blur_simu)
cv2.imshow("Gaussian Threshold", gaussian_thresh_simu)
cv2.imshow("Otsu Threshold", otsu_thresh_simu)
cv2.imshow("Final Filter", final_filter_simu)
cv2.imshow("Canny", canny_simu)
cv2.imshow("Hough", hough_simu)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Save
cv2.imwrite(saving_dic_simu+ "cropped_simu.jpg", cropped_simu)
cv2.imwrite(saving_dic_simu+ "gaussian_blur_simu.jpg", gaussian_blur_simu)
cv2.imwrite(saving_dic_simu + "gaussian_thresh_simu.jpg", gaussian_thresh_simu)
cv2.imwrite(saving_dic_simu + "otsu_thresh_simu.jpg", otsu_thresh_simu)
cv2.imwrite(saving_dic_simu + "final_filter_simu.jpg", final_filter_simu)
cv2.imwrite(saving_dic_simu + "canny_simu.jpg", canny_simu)
cv2.imwrite(saving_dic_simu + "hough_simu.jpg", hough_simu)


