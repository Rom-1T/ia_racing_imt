# Preprocessing

There are 2 reasons why the model require preprocessing the input image of the camera. The first one is to simplify the image, thus, the model has less "parameters" to take into account for its learning. The second reason only applies when (a part of) the learning is taking place in the simulator, for obvious reasons, the simulator does not look alike the reality. So, if the model is intended to be integrated into the reality, it require to have similar input images.

### Image cleaning

#### Gaussian blur

This is a common first step as gaussian blur usually typically to reduce image noise and reduce detail. This step is very important in real camera image as the camera create a lot of noise.

![Original img](img\_example/example\_ori.jpg.jpg) ![Gaussian blur](img\_example/gaussian\_blur.jpg)

#### Enhancing image features

For a faster learning, the AI model can be imputed with images with enhanced features. In the case of donkeycar, enhancing line and road is quite intuitive. There are a lot of widely known segmentation tool for edge detecting (Zero-Cross, Canny,...). Those techniques were not adapted to the project. Instead, we have chosen to use two adaptive threshold combined.

#### Edge detecting - Gaussian adaptive thresholding

The threshold value is the weighted sum of neighbourhood values where weights are a gaussian window. In our case, this techique enhance lines because the are the frontier between a black element (the road) and a white element (the lines). From experience, this method gives us better result that Laplace edge detecting because because the obtained image is less complex with wider lines. The problem is that we lost a lot of information. In fact this technique is good to detect the border of the path but as it suppress a lot of information, the model could be lost easily.

![Gaussian blur](img\_example/gaussian\_blur.jpg) ![Gaussian thresholding](img\_example/gaussian\_threshold.jpg)

#### Otsu adaptive thresholding

Otsu’s method is an adaptive thresholding way for binarization in image processing. It can find the optimal threshold value of the input image by going through all possible threshold values (from 0 to 255). In our case, we needed to binarize the input image as the road is black and the line are white. Which mean otsu will to separate two related data, for example, the road and the lines. The problem of this method is that it also enhance details and noise as everything become either completely black or completly white. For example, a light reflection on the road which were not very bright in the original image become as white as the lines (255)

![Gaussian blur](img\_example/gaussian\_blur.jpg) ![Otsu thresholding](img\_example/otsu\_treshold.jpg)

#### Final filter

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

This filter might require an optimization in the choice of the parameter in the gaussian threshold.

#### Canny method

The Canny Transform is a multi-step algorithm&#x20;

#### Hough Transform

#### Integration

For the integration, we choose to not choose. We are training our models with variation of each of these filters.&#x20;

* The "edge" filter is basically a Canny Algorithmein which the lines are dilated. This "dilatation" can be modulated with the variable "epaisseur".
* The "lines" filter is a filter based on the Hough Transform. It is a additional layer to the "edge" filter. It simplifies the image again to only prompt lines.
* The "final\_filter" is the one that combine the 2 adaptative threshold. It shows good results in reality but is way to sensitive to noises and color changes.



### Degrading images

#### Motion blur for the training

In the simulator and only in the simulator we can add a random motion blur so the input images are degraded which make them closer to real images.

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

This Python library contains different transforms for image augmentation. For a more robust model, we choose to create a function that is applied to the input image. This function will randomly applies one or more transform to the input image.&#x20;



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

The rain effect has been deleted because it was too hard for the model. This function can be tuned by modifying the two values "x" and "y". Increase them will make the degradation harder for the model. This function also contains a motion blur.

####
