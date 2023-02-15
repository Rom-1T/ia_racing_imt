# Preprocessing

There are 2 reasons why the model require preprocessing the input image of the camera. The first one is to simplify the image, thus, the model has less "parameters" to take into account for its learning. The second reason only applies when (a part of) the learning is taking place in the simulator, for obvious reasons, the simulator does not look alike the reality. So, if the model is intended to be integrated into the reality, it require to have similar input images. &#x20;

### Image cleaning

#### Gaussian blur

This is a common first step as gaussian blur usually typically to reduce image noise and reduce detail. This step is very important in real camera image as the camera create a lot of noise.

![Original img](img_example/example_ori.jpg.jpg)
![Gaussian blur](img_example/gaussian_blur.jpg)

#### Enhancing image features

For a faster learning, the AI model can be imputed with images with enhanced features. In the case of donkeycar, enhancing line and road is quite intuitive. There are a lot of widely known segmentation tool for edge detecting (Zero-Cross, Canny,...). Those techniques were not adapted to the project. Instead, we have chosen to use two adaptive threshold combined.&#x20;

#### Edge detecting - Gaussian adaptive thresholding

The threshold value is the weighted sum of neighbourhood values where weights are a gaussian window. In our case, this techique enhance lines because the are the frontier between a black element (the road) and a white element (the lines). From experience, this method gives us better result that Laplace edge detecting because because the obtained image is less complex with wider lines. The problem is that we lost a lot of information. In fact this technique is good to detect the border of the path but as it suppress a lot of information, the model could be lost easily.

![Gaussian blur](img_example/gaussian_blur.jpg)
![Gaussian thresholding](img_example/gaussian_threshold.jpg)

#### Otsu adaptive thresholding

Otsu’s method is an adaptive thresholding way for binarization in image processing. It can find the optimal threshold value of the input image by going through all possible threshold values (from 0 to 255). In our case, we needed to binarize the input image as the road is black and the line are white. Which mean otsu will to separate two related data, for example, the road and the lines. The problem of this method is that it also enhance details and noise as everything become either completely black or completly white. For example, a light reflection on the road which were not very bright in the original image become as white as the lines (255)

![Gaussian blur](img_example/gaussian_blur.jpg)
![Otsu thresholding](img_example/otsu_treshold.jpg)

#### Final filter

The idea of the filter is to superpose a gaussian thresholding and an otsu thresholding to keep effective details while enhancing the border.

![Original img](img_example/example_ori.jpg.jpg)
![Final filter](img_example/final_filter.jpg)

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





####
