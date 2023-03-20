# Preprocessing

There are 2 reasons why the model requires preprocessing the input image of the camera. The first one is to simplify the image, thus, the model has fewer "parameters" to take into account for its learning. The second reason only applies when (a part of) the learning is taking place in the simulator, for obvious reasons, the simulator does not look like reality. So, if the model is intended to be integrated into reality, it requires to have similar input images.

### Image cleaning

#### Gaussian blur

This is a common first step as gaussian blur is usually typically to reduce image noise and reduce detail. This step is very important in real camera image as the camera creates a lot of noise.

![Original img](img\_example/example\_ori.jpg.jpg) ![Gaussian blur](img\_example/gaussian\_blur.jpg)

#### Enhancing image features

For faster learning, the AI model can be imputed with images with enhanced features. In the case of donkey cars, enhancing line and road is quite intuitive. There is a lot of widely known segmentation tool for edge detecting (Zero-Cross, Canny,...). Those techniques were not adapted to the project. Instead, we have chosen to use two adaptive thresholds combined.

#### Edge detecting - Gaussian adaptive thresholding

The threshold value is the weighted sum of neighborhood values where weights are a gaussian window. In our case, this technique enhances lines because they are the frontier between a black element (the road) and a white element (the lines). From experience, this method gives us better results than Laplace edge detecting because the obtained image is less complex with wider lines. The problem is that we lost a lot of information. In fact, this technique is good to detect the border of the path but as it suppresses a lot of information, the model could be lost easily.

![Gaussian blur](img\_example/gaussian\_blur.jpg) ![Gaussian thresholding](img\_example/gaussian\_threshold.jpg)

#### Otsu adaptive thresholding

Otsu’s method is an adaptive thresholding way for binarization in image processing. It can find the optimal threshold value of the input image by going through all possible threshold values (from 0 to 255). In our case, we needed to binarize the input image as the road is black and the line are white. This means otsu will separate two related data, for example, the road and the lines. The problem of this method is that it also enhances details and noise as everything becomes either completely black or completly white. For example, a light reflection on the road which was not very bright in the original image become as white as the lines (255)

![Gaussian blur](img\_example/gaussian\_blur.jpg) ![Otsu thresholding](img\_example/otsu\_treshold.jpg)

#### "Final filter" (Not final :o )

The idea of the filter is to superpose a gaussian thresholding and an otsu thresholding to keep effective details while enhancing the border.

![Original img](img\_example/example\_ori.jpg.jpg) ![Final filter](img\_example/final\_filter.jpg)

```
def final_filter(image) :
    image = cv2.GaussianBlur(image, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    filter = cv2.addWeighted(threshold,1,image_result,0.5,0.0)
    return(filter)

```

This filter might require optimization in the choice of the parameter in the gaussian threshold.

#### Canny method

The idea behind Canny Edge detection is to identify the edges in an image by finding the areas where there is a sharp change in intensity. In other words, the algorithm tries to identify the boundaries between different regions in an image where the pixel values change significantly.

The Canny Edge detection algorithm works in several stages. The first stage is to apply a Gaussian filter to the image to remove any noise. The next stage is to compute the gradient of the image using a Sobel filter. This helps to identify the areas where there is a rapid change in intensity.

The next step is to apply non-maximum suppression. This step involves identifying the local maxima in the gradient image and suppressing all other pixels that are not part of an edge.

The final step is to apply hysteresis thresholding. This involves selecting two threshold values, a high threshold and a low threshold. Any pixel in the gradient image that is above the high threshold is considered to be part of an edge. Any pixel that is below the low threshold is considered to be not part of an edge. Any pixel that is between the high and low thresholds is only considered to be part of an edge if it is connected to a pixel that is above the high threshold.

The result of Canny Edge detection is a binary image where the white pixels represent the edges in the original image

#### Hough Transform

The basic idea behind the Hough Transform is to convert an image in its pixel representation into a parameter space representation, where the parameters of the image features can be represented as points. For example, in the case of line detection, the Hough Transform can be used to represent all possible lines that can pass through a set of points in the image as points in a parameter space.

The Hough Transform works in the following way:

1. For each edge pixel in the image, calculate the set of possible parameters for the image feature of interest. For example, in the case of line detection, this would involve calculating the possible values of the slope and y-intercept of a line that could pass through the edge pixel.
2. For each possible set of parameters, increment a corresponding accumulator array. This effectively creates a histogram of the possible lines that could be present in the image.
3. Find the maximum value in the accumulator array, which corresponds to the most likely parameters of the image feature.
4. Convert the maximum value in the accumulator array back to image coordinates to identify the image feature.

In our solution, we apply the Hough Transform to the Canny output.

```
// Function that applies the Hough Transform
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
    return 
```

#### Integration

For the integration, we choose to not choose. We are training our models with variations of each of these filters.&#x20;

* The "edge" filter is basically a Canny algorithm in which the lines are dilated. This "dilatation" can be modulated with the variable "epaisseur". Although this filter can sometimes enhance noises (Line, crease on the road,...) which can be deleted by adding the "lines" layer.
* The "lines" filter is a filter based on the Hough Transform. It is an additional layer to the "edge" filter. It simplifies the image again to only prompt lines. This filter has 2 major flaws. Firstly it consumes a lot of resources, on a PC it is rather insignificant but on the Raspberry Pi, it can lead to major performance issues. For instance, it can take around 30-40ms the compute an image. This means that, with only the preprocessing, we can reach a max 25Hz. Secondly, this filter's bright side is the information reduction, making the learning faster for the neural model but it is also his biggest flaw because when the image is critical (A sharp turn for example) a lot of information (the boundary of the road) can be disastrous. If you can provide the model with a lot of images per second it might not be too harsh (This is why in the simulator this filter is really good) but without performance issues, we are not able to give enough images per second.
* The "final\_filter" is the one that combines the 2 adaptative thresholds. It shows good results in reality but is way too sensitive to noises and color changes.

```
```

### Degrading images

#### Motion blur for the training

In the simulator and only in the simulator we can add a random motion blur so the input images are degraded which makes them closer to real images.

```
def motion_blur_effect(img,kernel_size=None) :
    if kernel_size == None :
        kernel_size = randint(2,10)
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
```

![Original img](img\_example/example\_ori.jpg.jpg) ![Motion blur](img\_example/degraded\_img.jpg)

#### Use Albumentation

This Python library contains different transforms for image augmentation. For a more robust model, we choose to create a function that is applied to the input image. This function will randomly apply one or more transform to the input image.&#x20;



```
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
```

The rain effect has been deleted because it was too hard for the model. This function can be tuned by modifying the two values "x" and "y". Increasing them will make the degradation harder for the model. This function also contains a motion blur.

####
