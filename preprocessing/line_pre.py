import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


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


def filter_img(img=None, v_min=100, v_max=200, filter_type="gaussian"):
    pre_filter = cv2.GaussianBlur(img, (3, 3), 0)
    if filter_type == "gaussian":
        return pre_filter
    elif filter_type=="median":
        return cv2.medianBlur(img, 3)
    elif filter_type == "canny" :
        edges_1D = cv2.Canny(pre_filter, v_min, v_max)
        edges_3D = np.stack((edges_1D, edges_1D, edges_1D), axis=2)
        filter = cv2.addWeighted(pre_filter, 0.4, edges_3D, 0.6, 0)
        return filter
    elif filter_type =="laplacian" :
        edges = cv2.Laplacian(pre_filter,ddepth=cv2.CV_8U)
        filter = cv2.addWeighted(pre_filter, 0.2, edges, 1, 0)
        return filter


# create figure
fig = plt.figure(figsize=(10, 7))
# setting values to rows and column variables
rows = 3
columns = 3

cam_img = cv2.imread("data/gen_track_user_drv_right_lane/1429_cam-image_array_.jpg", cv2.IMREAD_COLOR)
fig.add_subplot(rows, columns, 1)
plt.imshow(cam_img, cmap='gray')
plt.title("Original")

fig.add_subplot(rows, columns, 2)
plt.imshow(crop_img(img=cam_img, suppress_shadow=False), cmap='gray')
plt.title("Cropped")

fig.add_subplot(rows, columns, 3)
plt.imshow(crop_img(img=cam_img), cmap='gray')
plt.title("Cropped without camera shadow")

start = time.time()
filtered = filter_img(crop_img(img=cam_img), filter_type="laplacian")
end = time.time()
elapsed = end - start
print(f'Temps d\'exécution quand cropped with gaussian: {elapsed:.2}ms')

start = time.time()
filtered_median = filter_img(crop_img(img=cam_img), filter_type="median")
end = time.time()
elapsed = end - start
print(f'Temps d\'exécution quand cropped with median: {elapsed:.2}ms')

start = time.time()
filtered_gaussian_non_crop = filter_img(img=cam_img, filter_type="gaussian")
end = time.time()
elapsed = end - start
print(f'Temps d\'exécution quand non cropped with gaussian: {elapsed:.2}ms')

start = time.time()
filtered_median_non_crop = filter_img(img=cam_img, filter_type="median")
end = time.time()
elapsed = end - start
print(f'Temps d\'exécution quand non cropped with median: {elapsed:.2}ms')

fig.add_subplot(rows, columns, 4)
plt.imshow(cv2.GaussianBlur(crop_img(img=cam_img), (5, 5), 0), cmap='gray')
plt.title("Gaussian blur")

fig.add_subplot(rows, columns, 5)
plt.imshow(cv2.medianBlur(crop_img(img=cam_img), 5), cmap='gray')
plt.title("Median blur")

fig.add_subplot(rows, columns, 6)
plt.imshow(filtered, cmap='gray')
plt.title("Filtered (with gaussian)")

fig.add_subplot(rows, columns, 7)
plt.imshow(filtered_median, cmap='gray')
plt.title("Filtered (with median)")
plt.show()

print("IMG SIZE", cam_img.shape)
print("CROPPED", crop_img(cam_img).shape)
print("FILTERED", filtered.shape)
print(filtered)


# build_filtre = cv2.GaussianBlur(building_color, (3,3), 0)
# edges = cv2.Canny(build_filtre,100,200)
def create_data_set():
    for i in range(1, 1000):
        file_name = f"data/gen_track_user_drv_right_lane/{i}_cam-image_array_.jpg"
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        cropped = crop_img(img=img)
        filter = filter_img(img=cropped, v_min=150, v_max=200, filter_type="gaussian")
        saving_path = f"canny/{i}_cam-image_array_.jpg"
        cv2.imwrite(saving_path, filter)
