# Drive Reinforcement

### Gym Environment&#x20;

Detailed documentation: [https://github.com/tawnkramer/gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar)

#### Add a new circuit

When you have created a new circuit "TestCircuit" you need to create a new Gym Env. There are 3 steps to do so

1. Create a new class in your _donkey\_env.py_ file. In the directory where donkey-gym is installed. Find the **gym-donkeycar/gym\_donkeycar/envs/donkey\_env.py.** In this file, at the bottom, create a new class.

```
// gym-donkeycar/gym_donkeycar/envs/donkey_env.py
class TestCircuitEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="TestCircuit", *args, **kwargs)
```

2. Now you need to register this new env. **Go to gym-donkeycar/gym\_donkeycar/\_\_init\_\_.py/.** You need to add two pieces of code.

```
// gym-donkeycar/gym_donkeycar/__init__.py/
from gym_donkeycar.envs.donkey_env import (
    AvcSparkfunEnv,
    CircuitLaunchEnv,
    GeneratedRoadsEnv,
    GeneratedTrackEnv,
    MiniMonacoEnv,
    MountainTrackEnv,
    RoboRacingLeagueTrackEnv,
    ThunderhillTrackEnv,
    WarehouseEnv,
    WarrenTrackEnv,
    WaveshareEnv,
    TestCircuitEnv #HERE ADD THE NAME OF THE DONKEY ENV YOU CREATED
)

...

reregister(id="donkey-test-circuit-v0", entry_point="gym_donkeycar.envs.donkey_env:TestCircuitEnv ") # HERE REGISTER THE NEW ENV
```

3. (If you are using RL-Baseline Zoo only). In the hyperparameter file, you need to add the registered name

<pre><code>// Hyperparams file

<strong>donkey-test-circuit-v0:
</strong>  policy: 'MlpPolicy'
  n_timesteps: !!float 7.5e7
  learning_rate: !!float 0.03
  delta_std: !!float 0.025
  n_delta: 40
  n_top: 30
  ...
</code></pre>

Now when you are calling the train function, you must use the registered name.

```
python train.py --algo tqc --env donkey-test-circuit-v0 --save-freq 100000 ... 
```

#### Modify the env parameters

To make the training faster, you might need to modify some parameters of the donkey env instead of creating a _Gym Wrapper._ Go to **gym-donkeycar/gym\_donkeycar/envs/donkey\_env.py.**

```
// Modify the default param

def supply_defaults(conf: Dict[str, Any]) -> None:
    """
    Update the config dictonnary
    with defaults when values are missing.
    :param conf: The user defined config dict,
        passed to the environment constructor.
    """
    defaults = [
        ("start_delay", 5.0),
        ("max_cte", 8.0), #MAX DISTANCE TO THE CENTER OF THE ROAD
        ("frame_skip", 1),
        ("cam_resolution", (120, 160, 3)),
        ("log_level", logging.INFO),
        ("host", "localhost"),
        ("port", 9091),
        ("steer_limit", 1.0), #CHOOSE 0.5
        ("throttle_min", 0.0),
        ("throttle_max", 1.0),#CHOOSE 0.2 at the beginning
    ]
```

#### Create a Gym Wrapper

Creating a Gym Wrapper for a Donkey Gym environment can be a useful way to modify the input data that is passed to the environment, as well as the output data that is returned by the environment. One example of a modification you might want to make to the input data is to crop the image to focus on a specific area of interest.

To create a Gym Wrapper that crops the input image in half, you can follow these general steps:

1. Define your Gym Wrapper class. This class should inherit from the gym.Wrapper class, which provides a standard interface for modifying the input and output data of a Gym environment. Your class should also define a constructor that takes the original environment as an argument.
2. Modify the observation space. Observation space need to have the same shape as the input data
3. Modify the input data. In your wrapper's step() and reset() method, you can modify the input data before it is passed to the underlying environment. In this case, you can crop the image using cropY() function.

```
// Some code
def cropY(img, px_from_top):
    return img[px_from_top:np.shape(img)[0], :, :]
    
class CropWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, crop_Y):
        assert isinstance(env.observation_space, gym.spaces.Box)
        super(CropWrapper, self).__init__(env)
        self.crop_Y = crop_Y
        print("CropY : ", self.crop_Y)
        print("epaisseur : ", self.epaisseur)
        raw_sensor_size = self.viewer.get_sensor_size()
        self.new_sensor_size = ((raw_sensor_size[0] - crop_Y), raw_sensor_size[1],1)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.new_sensor_size, dtype=np.uint8)

    def step(self, action):
        # print("step")
        self.nbr_img += 1
        obs, reward, done, info = self.env.step(action)
        raw_obs = obs[:, :, ::-1] #BGR TO RGB
        cropped = cropY(raw_obs,self.crop_Y)
        return cropped , reward, done, info

    def reset(self):
        obs = self.env.reset()
        raw_obs = obs[:, :, ::-1] #BGR TO RGB
        cropped = cropY(raw_obs,self.crop_Y)
        return cropped
```

### RL-Baseline3-Zoo framework

Detailed documentation: [https://github.com/DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

Tutorial: [https://www.youtube.com/watch?v=ngK33h00iBE\&t=4312s](https://www.youtube.com/watch?v=ngK33h00iBE\&t=4312s)

This is the framework we used to train and use the model. The biggest workload is tuning the hyperparameters and creating the _Gym Wrappers._

### Autoencoder

Detailed documentation: [https://github.com/araffin/aae-train-donkeycar/tree/feat/live-twitch-2](https://github.com/araffin/aae-train-donkeycar/tree/feat/live-twitch-2)

An autoencoder is a type of neural network that is commonly used for unsupervised learning. The basic idea behind an autoencoder is to learn a compressed representation of the input data that captures the most important information in the data.

An autoencoder consists of two main parts: an encoder and a decoder. The encoder takes the input data and transforms it into a compressed representation, typically a lower-dimensional vector or matrix. The decoder then takes this compressed representation and transforms it back into the original input data.

During training, the autoencoder is trained to minimize the difference between the input data and the reconstructed output data. This is typically done by minimizing the mean squared error or some other distance metric between the input and output data.

Here are a few reasons why we use an AE :&#x20;

1. Data Compression: Autoencoders can be used to compress the raw image data from your Donkey Car's camera into a lower-dimensional representation, which can reduce the amount of data that needs to be processed and transmitted during training and inference. This can be especially useful when working with limited computing resources or low-bandwidth communication channels.
2. Feature Extraction: The compressed representation learned by an autoencoder can also be used as a feature vector for training a downstream model, such as a classifier or a regression model. By learning a compact and informative representation of the input data, autoencoders can help improve the performance of these models, especially when working with high-dimensional data such as images.
3. Data Augmentation: Autoencoders can also be used to generate new data that is similar to the original input data. This can be useful for data augmentation, where the goal is to increase the size of the training dataset by generating new samples with small variations from the original data. This can help improve the generalization performance of your Donkey Car AI model by reducing overfitting to the limited training data.
4. Denoising: Autoencoders can also be used to denoise corrupted data by reconstructing the original signal from noisy or incomplete measurements. This can be especially useful in situations where the input data is subject to noise or other forms of corruption, such as in low-light or high-speed driving conditions.

### Integration in the raspberry pi

<figure><img src=".gitbook/assets/pi_process.jpg" alt=""><figcaption><p>Process of the raspberry pi simplfied</p></figcaption></figure>


### Pytorch part (integration)
The pytorch part makes it possible to use pytorch models in donleycar. However it is not very flexible and the only models accepted are autoencoder(described in a previous chapter) + drive (loaded as TQC with the RL-Baseline Zoo library).
To use it, here are the steps to follow:

 - Put the models as drive.zip and ae.pkl in mycar/models.
 - Make sure that the pytorch.py, the autoencoder.py and ae_config.py parts are in the donkeycar/parts folder
 - add `--type pytorch`and  `--path [Path to mycar/models]` to `the manage.py drive` command line. 
 - Then by selecting the full auto or auto steering mode in the browser interface, the model should start driving. 
 - Make sure to update the config file with all arguments that you will find usefull (history, reconstruction...)
 
### Pytorch part (description)

There are 3 classes defined in the Pytorch Part. 
#### CNN 
The CNN Class is an attempt at using RL baseline Zoo's CNN architecture and avoid using autoencoder. However it won't work with the gym environment because it's a continuous environment.

#### MLP 
The MLP class loads autoencoder and drive models, and combines them by calling the CombinedModel class.
The objects created with the MLP class have a run function that will be called when driving in auto mode.
```
def run(self, img_arr: np.ndarray) \  
        -> Tuple[Union[float, np.ndarray], ...]:  

```
If this condition is true, reconstructs the image with the autoencoder, so that the input of the drive model can be ploted. If true, the reconstructed image will be added to the 'outputs'.
```
 if self.reconstruct :  
        order_lst, reconstructed_image = self.model.forward(img_arr, self.throttle_history_val, reconstruct=True)  
    else :  
        order_lst = self.model.forward(img_arr, self.throttle_history_val) 
```
This is a postprocessing of the throttle.
```
order = order_lst[0]  
    if order[1] > 0.22 :  
        order[1] = 0.22  
  if order[1] < 0.17 :  
        order[1] += 0.05  
```
If this condition is true, stores the throttle history directly in the class object, and pass it as argument to the CombinedModel object's forward function.
``` 
    
  if self.throttle_history :  
        self.throttle_history_val = np.insert(self.throttle_history_val,0,order[1])[:-1]  
        print("THROTTLE HISTORY : ", self.throttle_history_val)  
    logger.info('model order', str(order[0]),str(order[1]))  
    if self.reconstruct :  
        return order[0],order[1],reconstructed_image  
    else :  
        return order[0], order[1]
```

#### CombinedModel
Creates a model that is a combinaison of an autoencoder and a drive model.
```
def __init__(self, model1, model2):  
        self.model1 = model1  
        self.model2 = model2  
```

Depending on throttle history input, concatenates the autoencoder output and the throttle history, before passing it as input to the drive model.
```
def forward(self, x, throttle_history=None, reconstruct=False):  
        encoded_img = self.model1.encode_from_raw_image(x)  
        if throttle_history is not None :  
            x1 = np.concatenate((encoded_img.flatten(),throttle_history))  
        else :  
            x1 = encoded_img.flatten()  
```

Depending on the reconstruct input, uses the decoder and returns the decoded image as output.

```
x2 = self.model2.predict(x1, deterministic=True)  
        if reconstruct :  
            reconstructed_image = self.model1.decode(encoded_img)[0]  
            return x2, reconstructed_image  
        else :  
            return x2
```
### Autoencoder (tips)
The autoencoder project is very simple to use but very difficult to modify without breaking it all.
We managed to add our own data augmentation by adding a noiser in the `_make_batch_element function` : 

```
def _make_batch_element(cls, image_path, augmenter=None, image_directory=None, noiser=None):  
    """  

									[...]
  
    if augmenter is not None:  
        input_img = augmenter.augment_image(  
            preprocess_image(im.copy(), normalize=False), hooks=imgaug.HooksImages(postprocessor=postprocessor)  
        )  
        # Normalize  
  input_img = augmenter(image=input_img)['image']  
        input_img = preprocess_input(input_img.astype(np.float32), mode="rl")  
        input_img = input_img.reshape((1,) + input_img.shape)  
    if image_directory is not None:  
        im = cv2.imread(image_path2)  
  
    if noiser is not None:  
        input_img = noiser(  
            preprocess_image(im.copy(), normalize=False))  
        # Normalize  
  input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)  
        input_img = preprocess_input(input_img.astype(np.float32), mode="rl")  
        input_img = input_img.reshape((1,) + input_img.shape)

```

You can also directly modify the already implemented `get_image_augmenter ` function:

```
def get_image_augmenter() -> iaa.Sequential:  
    """  
 :return: Image Augmenter """  return iaa.Sequential(  
        [  
            Sometimes(0.5, iaa.Fliplr(1)),  
  # Add shadows (from https://github.com/OsamaMazhar/Random-Shadows-Highlights)  
  Sometimes(0.3, RandomShadows(1.0)),  
  # Sometimes(0.3, iaa.MultiplyBrightness((0.8, 1.2))),  
  Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0))),  
  Sometimes(0.5, iaa.MotionBlur(k=(3, 11), angle=(0, 360))),  
  # Sometimes(0.5, iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))),  
  Sometimes(0.4, iaa.Add((-25, 25), per_channel=0.5)),  
  # Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),  
 # Sometimes(0.2, iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.10), per_channel=0.5)), # 20% of the corresponding size of the height and width  Sometimes(0.3, iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)),  
  # Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 1.8), per_channel=0.5)),  
 # Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=10, per_channel=True))  ],  
  random_order=True,  
  )
```
In both cases, make sure that the data fits the expectations of both the autoencoder (RGB format) and the restriction imposed by the autoencoder.py script:
` assert observation.shape == self.input_dimension, f"{observation.shape} != {self.input_dimension}" `). 
