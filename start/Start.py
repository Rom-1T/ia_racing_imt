import numpy as np
import cv2

################################################################################################################################################

# Specifying upper and lower ranges of color to detect in hsv format

# for green
lower_green = np.array([50, 50, 160])    # High value 
upper_green = np.array([60, 255, 255])   # Low value

# for red
lower_red = np.array([115, 50, 160])     # High value
upper_red = np.array([130, 255, 255])    # Low value



# Has the car to be launched?
launch_car=0

# Position on x and y, weight and height of the biggest red rectangle we've detected named BIG_RED_RECT
(x_red, y_red, w_red, h_red) = (-1,-1,0,0)

# We only accept a difference of 10% between each same parameter (betwenn both x, both y, ...)
max_diff = 0.1  

################################################################################################################################################

""" 
    Function called by the main program that is the center piece of START
    - INPUT: image, the image get by the camera
    - OUTPUT: NONE
"""
def getPicture(image):

    # Position on x and y, weight and high of the biggest red rectangle we've detected named BIG_RED_RECT
    global x_red, y_red, w_red, h_red

    # Lecture of the image with cv2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # We get shape of the image
    (img_height, img_width,_) = image.shape                                                 

    ############### WE CHECK RED RECTANGLE ###############
    x, y, w, h = getRedRectangle(image)


    ############### IF IT'S NOT THE SAME RECTANGLE THAN BIG_RED_RECT ###############
    if ( IsSameRectangle (x_red, y_red, w_red, h_red, x, y , w, h, img_height, img_width) == False):
        
        # If it's a bigger red rectangle
        if (w*h > w_red*h_red):
            x_red, y_red, w_red, h_red = x, y, w, h                                # We save these parameters of red rectangle
        

    ################ WE CHECK GREEN RECTANGLE ################
    x, y, w, h = getGreenRectangle(image)

    ############### IF IT'S THE SAME RECTANGLE THAN BIG_RED_RECT ###############
    if ( IsSameRectangle (x_red, y_red, w_red, h_red, x, y, w, h, img_height, img_width) == True):
        launch_car = True

################################################################################################################################################

""" 
    Function called by getPicture to keep the above part of the image
    - INPUT: image, the image get by the camera
    - OUTPUT: the image cropped
"""

def cutPicture(image):

    # We get shape of the image and the half the height
    (height,_,_) = image.shape                                                 
    half_height = height//2
    
    # We only keep the above part of the image
    image = image[:half_height, :]

    return image

################################################################################################################################################

""" 
    Function called by getPicture to get the x and y positions, the weight and the high of the biggest red rectangle detected
    - INPUT: image, the image get by the camera cropped
    - OUTPUT: x,y,h,w the x and y positions, the weight and the high of the biggest red rectangle detected
"""

def getRedRectangle(image):

    # We initialise x,y,w and h corresponding to parameters of the biggest red rectangle we found on this image
    x, y= -1,-1
    w, h = 0,0

    # We get parameters to detect red color
    global lower_red, upper_red

    # We treat the image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Converting BGR image to HSV format
    img = cv2.blur(img, (5,5))  # Blur
    mask = cv2.inRange(img, lower_red, upper_red) # Masking the image to find our color

    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=4)
    image2 = cv2.bitwise_and(image, image, mask=mask)

    # We find all red rectangles
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image

    ################ WE BROWSE ALL RECTANGLES ################
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:

            # We get x,y,w,h
            x1, y1, w1, h1 = cv2.boundingRect(mask_contour)

            # If this rectangle is bigger than the one we found 
            if (w1*h1 > w*h):
                x, y, w, h = x1, y1, w1, h1

    # We return x,y,w and h of the biggest red rectangle we found on this image
    return x, y, w, h

################################################################################################################################################

""" 
    Function called by getPicture to get the x and y positions, the weight and the high of the biggest green rectangle detected
    - INPUT: image, the image get by the camera cropped
    - OUTPUT: x,y,h,w the x and y positions, the weight and the high of the biggest green rectangle detected
"""

def getGreenRectangle(image):

    # We initialise x,y,w and h corresponding to parameters of the biggest green rectangle we found on this image
    x, y = -1,-1
    w, h = 0,0

    # We get parameters to detect green color
    global lower_green, upper_green

    # We treat the image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Converting BGR image to HSV format
    img = cv2.blur(img, (5,5))  # Blur
    mask = cv2.inRange(img, lower_green, upper_green) # Masking the image to find our color

    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=4)
    image2 = cv2.bitwise_and(image, image, mask=mask)

    # We find all green rectangles
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image

    ################ WE BROWSE ALL RECTANGLES ################
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:

            # We get x,y,w,h
            x1, y1, w1, h1 = cv2.boundingRect(mask_contour)

            # If this rectangle is bigger than the one we found 
            if (w1*h1 > w*h):
                x, y, w, h = x1, y1, w1, h1

    # We return x,y,w and h of the biggest green rectangle we found on this image
    return x, y, w, h

################################################################################################################################################

""" 
    Function called by getPicture to get the x and y positions, the weight and the high of the biggest green rectangle detected
    - INPUT: image, the image get by the camera cropped
    - OUTPUT: true if we assume they are the same rectangle, false otherwise
"""

def IsSameRectangle(x1, y1, w1, h1, x2, y2, w2, h2, img_height, img_width):

    # We get the maximum difference between two same rectangle we want
    global max_diff

    # We calcule the maximum difference we have between the four same parameters
    max_diff_mes = max( abs((x2 - x1)/img_width), abs((y2 - y1)/img_height), abs((w2 - w1)/img_height), abs((h2 - h1)/img_width) )

    # We say if this maximum difference we have is small enough or not
    return max_diff_mes < max_diff

################################################################################################################################################

