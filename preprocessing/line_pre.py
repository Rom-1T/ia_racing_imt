import numpy as np
import matplotlib.pyplot as plt
import cv2

building_color = cv2.imread("building.jpg",cv2.IMREAD_GRAYSCALE)
build_filtre = cv2.GaussianBlur(building_color, (3,3), 0)
edges = cv2.Canny(build_filtre,100,200)