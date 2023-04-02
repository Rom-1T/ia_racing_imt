#!/usr/bin/env python

from picamera import PiCamera
from time import sleep
import os

path= "/log"
try:
    os.mkdir(path)
except OSError as error:
    pass
resolution = (160,120)
camera = PiCamera()
camera.resolution = resolution
camera.start_preview()
sleep(5)
for i in range (0,50) :
    camera.capture(path +f'/fisheye_{i}.jpg')
camera.stop_preview()
