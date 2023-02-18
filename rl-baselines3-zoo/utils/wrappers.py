import os
import random
from copy import deepcopy
from typing import Optional

import gym
import numpy as np
import torch as th
from sb3_contrib.common.wrappers import TimeFeatureWrapper  # noqa: F401 (backward compatibility)
from scipy.signal import iirfilter, sosfilt, zpk2sos
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
import cv2
#from deformation import motion_blur_effect
#from filter import final_filter
import time
import matplotlib.pyplot as plt


class VecForceResetWrapper(VecEnvWrapper):
    """
    For all environments to reset at once,
    and tell the agent the trajectory was truncated.

    :param venv: The vectorized environment
    """

    def __init__(self, venv: VecEnv):
        super().__init__(venv=venv)

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            self._save_obs(env_idx, obs)

        if self.buf_dones.any():
            for env_idx in range(self.num_envs):
                self.buf_infos[env_idx]["terminal_observation"] = self.buf_obs[None][env_idx]
                if not self.buf_dones[env_idx]:
                    self.buf_infos[env_idx]["TimeLimit.truncated"] = True
                self.buf_dones[env_idx] = True
                obs = self.envs[env_idx].reset()
                self._save_obs(env_idx, obs)

        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """

    def __init__(self, env: gym.Env, reward_offset: float = 0.0, n_successes: int = 1):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset
        self.n_successes = n_successes
        self.current_successes = 0

    def reset(self):
        self.current_successes = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info.get("is_success", False):
            self.current_successes += 1
        else:
            self.current_successes = 0
        # number of successes in a row
        done = done or self.current_successes >= self.n_successes
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class ActionNoiseWrapper(gym.Wrapper):
    """
    Add gaussian noise to the action (without telling the agent),
    to test the robustness of the control.

    :param env: (gym.Env)
    :param noise_std: (float) Standard deviation of the noise
    """

    def __init__(self, env, noise_std=0.1):
        super(ActionNoiseWrapper, self).__init__(env)
        self.noise_std = noise_std

    def step(self, action):
        noise = np.random.normal(np.zeros_like(action), np.ones_like(action) * self.noise_std)
        noisy_action = action + noise
        return self.env.step(noisy_action)


# from https://docs.obspy.org
def lowpass(data, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Lowpass Filter.

    Filter data removing data over certain frequency ``freq`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = "Selected corner frequency is above Nyquist. " + "Setting Nyquist as high corner."
        print(msg)
    z, p, k = iirfilter(corners, f, btype="lowpass", ftype="butter", output="zpk")
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


class LowPassFilterWrapper(gym.Wrapper):
    """
    Butterworth-Lowpass

    :param env: (gym.Env)
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    """

    def __init__(self, env, freq=5.0, df=25.0):
        super(LowPassFilterWrapper, self).__init__(env)
        self.freq = freq
        self.df = df
        self.signal = []

    def reset(self):
        self.signal = []
        return self.env.reset()

    def step(self, action):
        self.signal.append(action)
        filtered = np.zeros_like(action)
        for i in range(self.action_space.shape[0]):
            smoothed_action = lowpass(np.array(self.signal)[:, i], freq=self.freq, df=self.df)
            filtered[i] = smoothed_action[-1]
        return self.env.step(filtered)




class DelayedRewardWrapper(gym.Wrapper):
    """
    Delay the reward by `delay` steps, it makes the task harder but more realistic.
    The reward is accumulated during those steps.

    :param env: (gym.Env)
    :param delay: (int) Number of steps the reward should be delayed.
    """

    def __init__(self, env, delay=10):
        super(DelayedRewardWrapper, self).__init__(env)
        self.delay = delay
        self.current_step = 0
        self.accumulated_reward = 0.0

    def reset(self):
        self.current_step = 0
        self.accumulated_reward = 0.0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.accumulated_reward += reward
        self.current_step += 1

        if self.current_step % self.delay == 0 or done:
            reward = self.accumulated_reward
            self.accumulated_reward = 0.0
        else:
            reward = 0.0
        return obs, reward, done, info


class HistoryWrapper(gym.Wrapper):
    """
    Stack past observations and actions to give an history to the agent.

    :param env: (gym.Env)
    :param horizon: (int) Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 5):
        assert isinstance(env.observation_space, gym.spaces.Box)

        wrapped_obs_space = env.observation_space
        wrapped_action_space = env.action_space

        # TODO: double check, it seems wrong when we have different low and highs
        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)

        low_action = np.repeat(wrapped_action_space.low, horizon, axis=-1)
        high_action = np.repeat(wrapped_action_space.high, horizon, axis=-1)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)

        super(HistoryWrapper, self).__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self):
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs = self.env.reset()
        self.obs_history[..., -obs.shape[-1]:] = obs
        return self._create_obs_from_history()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1]:] = action
        return self._create_obs_from_history(), reward, done, info


class HistoryWrapperObsDict(gym.Wrapper):
    """
    History Wrapper for dict observation.

    :param env: (gym.Env)
    :param horizon: (int) Number of steps to keep in the history.
    """

    def __init__(self, env, horizon=5):
        assert isinstance(env.observation_space.spaces["observation"], gym.spaces.Box)

        wrapped_obs_space = env.observation_space.spaces["observation"]
        wrapped_action_space = env.action_space

        # TODO: double check, it seems wrong when we have different low and highs
        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)

        low_action = np.repeat(wrapped_action_space.low, horizon, axis=-1)
        high_action = np.repeat(wrapped_action_space.high, horizon, axis=-1)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space.spaces["observation"] = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)

        super(HistoryWrapperObsDict, self).__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self):
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs_dict = self.env.reset()
        obs = obs_dict["observation"]
        self.obs_history[..., -obs.shape[-1]:] = obs

        obs_dict["observation"] = self._create_obs_from_history()

        return obs_dict

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = obs_dict["observation"]
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1]:] = action

        obs_dict["observation"] = self._create_obs_from_history()

        return obs_dict, reward, done, info


class ResidualExpertWrapper(gym.Wrapper):
    """
    :param env:
    :param model_path:
    :param add_expert_to_obs:
    :param residual_scale:
    """

    def __init__(
            self,
            env: gym.Env,
            model_path: Optional[str] = os.environ.get("MODEL_PATH"),
            add_expert_to_obs: bool = True,
            residual_scale: float = 0.2,
            expert_scale: float = 1.0,
            d3rlpy_model: bool = False,
    ):
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert model_path is not None

        wrapped_obs_space = env.observation_space

        low = np.concatenate((wrapped_obs_space.low, np.finfo(np.float32).min * np.ones(2)))
        high = np.concatenate((wrapped_obs_space.high, np.finfo(np.float32).max * np.ones(2)))

        # Overwrite the observation space
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)

        super(ResidualExpertWrapper, self).__init__(env)

        print(f"Loading model from {model_path}")
        if d3rlpy_model:
            self.model = th.jit.load(model_path)
        else:
            self.model = SAC.load(model_path)
        self.d3rlpy_model = d3rlpy_model
        self._last_obs = None
        self.residual_scale = residual_scale
        self.expert_scale = expert_scale
        self.add_expert_to_obs = add_expert_to_obs

    def _predict(self, obs):
        # TODO: move to gpu when possible
        if self.d3rlpy_model:
            expert_action = self.model(th.tensor(obs).reshape(1, -1)).cpu().numpy()[0, :]
        else:
            expert_action, _ = self.model.predict(obs, deterministic=True)
        if self.add_expert_to_obs:
            obs = np.concatenate((obs, expert_action), axis=-1)
        return obs, expert_action

    def reset(self):
        obs = self.env.reset()
        obs, self.expert_action = self._predict(obs)
        return obs

    def step(self, action):
        action = np.clip(self.expert_scale * self.expert_action + self.residual_scale * action, -1.0, 1.0)
        obs, reward, done, info = self.env.step(action)
        obs, self.expert_action = self._predict(obs)

        return obs, reward, done, info


def crop_img(img, crop_ratio):
    ratio = crop_ratio / 100
    nb_line_keep = int(len(img) * ratio)
    new_img = img[(int(len(img) - nb_line_keep)):int(len(img) * 0.9)]
    return new_img


def filter_img(img=None, v_min=100, v_max=200, filter_type="gaussian", nbr_img=0, random_nbr=0, alpha=0.4, beta=0.6,
               log_activated=False):
    pre_filter = cv2.GaussianBlur(img, (3, 3), 0)
    if filter_type == "gaussian_final":
        pre_filter = motion_blur_effect(img)
        filter = final_filter(pre_filter)
    elif filter_type == "median":
        filter = cv2.medianBlur(img, 3)
    elif filter_type == "canny":
        edges_1D = cv2.Canny(pre_filter, v_min, v_max)
        edges_3D = np.stack((edges_1D, edges_1D, edges_1D), axis=2)
        filter = cv2.addWeighted(pre_filter, alpha, edges_3D, beta, 0)
    elif filter_type == "laplacian":
        edges = cv2.Laplacian(pre_filter, ddepth=cv2.CV_8U)
        filter = cv2.addWeighted(pre_filter, alpha, edges, beta, 0)
    elif filter_type == "sobel":
        filter = cv2.Sobel(pre_filter, ddepth=cv2.CV_64F, dx=1, dy=0)
    elif filter_type == "erode":
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(pre_filter, kernel)
        filter = cv2.dilate(eroded, kernel)
    elif filter_type == "erode+laplacian":
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(pre_filter, kernel)
        filter = cv2.dilate(eroded, kernel)
        filter = cv2.Laplacian(filter, ddepth=cv2.CV_64F)
    elif filter_type == "laplacian+erode":
        pre_filter = cv2.Laplacian(pre_filter, ddepth=cv2.CV_64F)
        kernel = np.ones((5, 5), np.uint8)
        filter = cv2.dilate(pre_filter, kernel)
    elif filter_type == "log":
        filter = img
    else:
        filter_type = "none"
        filter = pre_filter
    if log_activated:
        directory_path = f"log_img/log_{filter_type}_{random_nbr}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        img_to_save = f"/img_{nbr_img}.jpg"
        path = directory_path + img_to_save
        cv2.imwrite(path, filter)
    return filter


class PreProcessingWrapper(gym.Wrapper):
    """
    PreProcess the img received by the camera for better learning rate

    :param env: (gym.Env)
    :param crop_ratio : int
    :param filter_type: string
    :param min_luminosity_value: int
    :param max_luminosity_value: int
    """

    def __init__(self, env: gym.Env, crop_ratio=60, filter_type="gaussian_final", min_luminosity_value=150,
                 max_luminosity_value=200, alpha=0.4, beta=0.6, log_activated=False, normalize=True):
        assert isinstance(env.observation_space, gym.spaces.Box)
        super(PreProcessingWrapper, self).__init__(env)
        self.filter_type = filter_type
        self.crop_ratio = crop_ratio
        self.min_luminosity_value = min_luminosity_value
        self.max_luminosity_value = max_luminosity_value
        self.alpha = alpha
        self.beta = beta
        self.nbr_img = 0
        self.log_activated = log_activated
        self.random_nbr = random.randint(0, 10000)
        self.normalize = normalize
        # Overwrite the observation space
        raw_sensor_size = self.viewer.get_sensor_size()
        # new_sensor_size = (int(raw_sensor_size[0] * (crop_ratio / 100 - 0.1)), raw_sensor_size[1], raw_sensor_size[2])
        if self.normalize:
            new_sensor_size = (
                int(raw_sensor_size[0] * (crop_ratio / 100 - 0.1)) * raw_sensor_size[1] * raw_sensor_size[2],)
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=new_sensor_size, dtype=np.float32)
        else:
            new_sensor_size = (int(raw_sensor_size[0] * (crop_ratio / 100 - 0.1)),raw_sensor_size[1],raw_sensor_size[2])
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_sensor_size, dtype=np.uint8)

    def step(self, action):
        self.nbr_img += 1
        obs, reward, done, info = self.env.step(action)
        processed_obs = filter_img(crop_img(obs[:, :, ::-1], self.crop_ratio), self.min_luminosity_value,
                                   self.max_luminosity_value,
                                   filter_type=self.filter_type, nbr_img=self.nbr_img, random_nbr=self.random_nbr,
                                   alpha=self.alpha, beta=self.beta, log_activated=self.log_activated)
        if self.normalize:
            processed_obs = processed_obs.flatten()
            return processed_obs / 255, reward, done, info
        return processed_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        processed_obs = filter_img(crop_img(obs[:, :, ::-1], self.crop_ratio), self.min_luminosity_value,
                                   self.max_luminosity_value,
                                   filter_type=self.filter_type, nbr_img=self.nbr_img, random_nbr=self.random_nbr,
                                   alpha=self.alpha, beta=self.beta, log_activated=self.log_activated)
        if self.normalize:
            processed_obs = processed_obs.flatten()
            return processed_obs / 255
        return processed_obs


class SteeringSmoothingWrapper(gym.Wrapper):
    def __init__(self, env, smoothing_coef: float = 0.0):
        super(SteeringSmoothingWrapper, self).__init__(env)
        self.past_action = None
        self.smoothing_coef = smoothing_coef

    def reset(self):
        self.past_action = None
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.past_action is None:
            self.past_action = np.zeros_like(action)
        else:
            reward_loss = self.smoothing_coef * (action[0]-self.past_action[0])/action[0]
            reward = reward - abs(reward_loss * reward)
            if reward < -1.0 :
                reward = -1.0
            self.past_action = action
        return obs, reward, done, info

class ActionSmoothingWrapper(gym.Wrapper):
    """
    Smooth the action using exponential moving average.

    :param env: (gym.Env)
    :param smoothing_coef: (float) Smoothing coefficient (0 no smoothing, 1 very smooth)
    """

    def __init__(self, env, smoothing_coef: float = 0.0):
        super(ActionSmoothingWrapper, self).__init__(env)
        self.smoothing_coef = smoothing_coef
        self.smoothed_action = None
        # from https://github.com/rail-berkeley/softlearning/issues/3
        # for smoothing latent space
        # self.alpha = self.smoothing_coef
        # self.beta = np.sqrt(1 - self.alpha ** 2) / (1 - self.alpha)

    def reset(self):
        self.smoothed_action = None
        return self.env.reset()

    def step(self, action):
        if self.smoothed_action is None:
            self.smoothed_action = np.zeros_like(action)
        self.smoothed_action = self.smoothing_coef * self.smoothed_action + (1 - self.smoothing_coef) * action
        return self.env.step(self.smoothed_action)


class PastThrottle(gym.Wrapper) :
    def __init__(self, env: gym.Env,length=5,auto_encoder_size = 32):
        super().__init__(env)
        self.autoencodersize = auto_encoder_size
        self.length = length
        self.past_throttle = np.zeros(self.length)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.length + self.autoencodersize,),
            dtype=np.float32,
        )
    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self.past_throttle = np.zeros(self.length)
        new_obs = np.concatenate([obs, self.past_throttle])
        return new_obs.flatten()

    def step(self, action: np.ndarray):
        obs, reward, done, infos = self.env.step(action)
        last_throttle = action[1]
        self.past_throttle = np.insert(self.past_throttle, 0, last_throttle, axis=0)
        self.past_throttle = np.delete(self.past_throttle,self.length)
        new_obs = np.concatenate([obs, self.past_throttle])

        return new_obs.flatten(), reward, done, infos

class Deadzone(gym.Wrapper) :
    def __init__(self, env: gym.Env,deadzone = 0.15):
        super().__init__(env)
        self.deadzone = deadzone
    def step(self, action: np.ndarray):
        if action[1] <= self.deadzone :
            action[1] = 0
        return self.env.step(action)
