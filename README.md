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
