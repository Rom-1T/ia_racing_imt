import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def line_detection_non_vectorized(image, edge_image, num_rhos=180, num_thetas=180, t_count=220):
  edge_height, edge_width = edge_image.shape[:2]
  edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
  #
  d = np.sqrt(np.square(edge_height) + np.square(edge_width))
  dtheta = 180 / num_thetas
  drho = (2 * d) / num_rhos
  #
  thetas = np.arange(0, 180, step=dtheta)
  rhos = np.arange(-d, d, step=drho)
  #
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  #
  accumulator = np.zeros((len(rhos), len(rhos)))
  #
  figure = plt.figure(figsize=(12, 12))
  subplot1 = figure.add_subplot(1, 4, 1)
  subplot1.imshow(image)
  subplot2 = figure.add_subplot(1, 4, 2)
  subplot2.imshow(edge_image, cmap="gray")
  subplot3 = figure.add_subplot(1, 4, 3)
  subplot3.set_facecolor((0, 0, 0))
  subplot4 = figure.add_subplot(1, 4, 4)
  subplot4.imshow(image)
  #
  for y in range(edge_height):
    for x in range(edge_width):
      if edge_image[y][x] != 0:
        edge_point = [y - edge_height_half, x - edge_width_half]
        ys, xs = [], []
        for theta_idx in range(len(thetas)):
          rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
          theta = thetas[theta_idx]
          rho_idx = np.argmin(np.abs(rhos - rho))
          accumulator[rho_idx][theta_idx] += 1
          ys.append(rho)
          xs.append(theta)
        subplot3.plot(xs, ys, color="white", alpha=0.05)

  for y in range(accumulator.shape[0]):
    for x in range(accumulator.shape[1]):
      if accumulator[y][x] > t_count:
        rho = rhos[y]
        theta = thetas[x]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + edge_width_half
        y0 = (b * rho) + edge_height_half
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        subplot3.plot([theta], [rho], marker='o', color="yellow")
        subplot4.add_line(mlines.Line2D([x1, x2], [y1, y2]))

  subplot3.invert_yaxis()
  subplot3.invert_xaxis()

  subplot1.title.set_text("Original Image")
  subplot2.title.set_text("Edge Image")
  subplot3.title.set_text("Hough Space")
  subplot4.title.set_text("Detected Lines")
  plt.show()
  return accumulator, rhos, thetas

DIM=(160, 120)
K=np.array([[76.40210369377033, 0.0, 85.17741324657462], [0.0, 75.55575570884872, 61.5111216120113], [0.0, 0.0, 1.0]])
D=np.array([[0.032858036745614], [-0.09739958496116238], [0.07344214252074698], [-0.02977154953395648]])


def lines(img,edges):
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
    #print("UNDISTORT")
    #img_array = np.array(img_array)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img_array, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def cropY(img, px_from_top):
    return img[px_from_top:np.shape(img)[0], :, :]

if __name__ == "__main__":
  for i in range(1,1000):
    img = cv2.imread(f"data/log_data_test_0/img_{i}.jpg")
    image = cropY(img, 40)
    edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.GaussianBlur(edge_image, (3, 3), 1)
    edge_image = cv2.Canny(edge_image, 150, 200, apertureSize=3)
    edge_image = cv2.dilate(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2
    )
    edge_image = cv2.erode(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2
    )
    cv2.imshow("original", img)
    cv2.imshow("cropped", image)
    cv2.imshow("edge", edge_image)
    cv2.imshow("lines", lines(image,edge_image))
    cv2.waitKey()
#
# def crop_img(img=None, crop_pourcent=75, keep="down", suppress_shadow=True):
#     ratio = crop_pourcent / 100
#     nb_line_keep = int(len(img) * ratio)
#     if keep == "down":
#         if suppress_shadow:
#             new_img = img[(int(len(img) - nb_line_keep)):int(len(img) * 0.9)]
#         else:
#             new_img = img[(int(len(img) - nb_line_keep)):]
#     else:
#         new_img = img[:nb_line_keep]
#     return new_img
#
#
# def filter_img(img=None, v_min=100, v_max=200, filter_type="gaussian"):
#     pre_filter = cv2.GaussianBlur(img, (3, 3), 0)
#     if filter_type == "gaussian":
#         return pre_filter
#     elif filter_type=="median":
#         return cv2.medianBlur(img, 3)
#     elif filter_type == "canny" :
#         edges_1D = cv2.Canny(pre_filter, v_min, v_max)
#         edges_3D = np.stack((edges_1D, edges_1D, edges_1D), axis=2)
#         filter = cv2.addWeighted(pre_filter, 0.4, edges_3D, 0.6, 0)
#         return filter
#     elif filter_type =="laplacian" :
#         edges = cv2.Laplacian(pre_filter,ddepth=cv2.CV_8U)
#         filter = cv2.addWeighted(pre_filter, 0.2, edges, 1, 0)
#         return filter
#
#
# # create figure
# fig = plt.figure(figsize=(10, 7))
# # setting values to rows and column variables
# rows = 3
# columns = 3
#
# cam_img = cv2.imread("data/gen_track_user_drv_right_lane/1429_cam-image_array_.jpg", cv2.IMREAD_COLOR)
# fig.add_subplot(rows, columns, 1)
# plt.imshow(cam_img, cmap='gray')
# plt.title("Original")
#
# fig.add_subplot(rows, columns, 2)
# plt.imshow(crop_img(img=cam_img, suppress_shadow=False), cmap='gray')
# plt.title("Cropped")
#
# fig.add_subplot(rows, columns, 3)
# plt.imshow(crop_img(img=cam_img), cmap='gray')
# plt.title("Cropped without camera shadow")
#
# start = time.time()
# filtered = filter_img(crop_img(img=cam_img), filter_type="laplacian")
# end = time.time()
# elapsed = end - start
# print(f'Temps d\'exécution quand cropped with gaussian: {elapsed:.2}ms')
#
# start = time.time()
# filtered_median = filter_img(crop_img(img=cam_img), filter_type="median")
# end = time.time()
# elapsed = end - start
# print(f'Temps d\'exécution quand cropped with median: {elapsed:.2}ms')
#
# start = time.time()
# filtered_gaussian_non_crop = filter_img(img=cam_img, filter_type="gaussian")
# end = time.time()
# elapsed = end - start
# print(f'Temps d\'exécution quand non cropped with gaussian: {elapsed:.2}ms')
#
# start = time.time()
# filtered_median_non_crop = filter_img(img=cam_img, filter_type="median")
# end = time.time()
# elapsed = end - start
# print(f'Temps d\'exécution quand non cropped with median: {elapsed:.2}ms')
#
# fig.add_subplot(rows, columns, 4)
# plt.imshow(cv2.GaussianBlur(crop_img(img=cam_img), (5, 5), 0), cmap='gray')
# plt.title("Gaussian blur")
#
# fig.add_subplot(rows, columns, 5)
# plt.imshow(cv2.medianBlur(crop_img(img=cam_img), 5), cmap='gray')
# plt.title("Median blur")
#
# fig.add_subplot(rows, columns, 6)
# plt.imshow(filtered, cmap='gray')
# plt.title("Filtered (with gaussian)")
#
# fig.add_subplot(rows, columns, 7)
# plt.imshow(filtered_median, cmap='gray')
# plt.title("Filtered (with median)")
# plt.show()
#
# print("IMG SIZE", cam_img.shape)
# print("CROPPED", crop_img(cam_img).shape)
# print("FILTERED", filtered.shape)
# print(filtered)
#
#
# # build_filtre = cv2.GaussianBlur(building_color, (3,3), 0)
# # edges = cv2.Canny(build_filtre,100,200)
# def create_data_set():
#     for i in range(1, 1000):
#         file_name = f"data/gen_track_user_drv_right_lane/{i}_cam-image_array_.jpg"
#         img = cv2.imread(file_name, cv2.IMREAD_COLOR)
#         cropped = crop_img(img=img)
#         filter = filter_img(img=cropped, v_min=150, v_max=200, filter_type="gaussian")
#         saving_path = f"canny/{i}_cam-image_array_.jpg"
#         cv2.imwrite(saving_path, filter)
