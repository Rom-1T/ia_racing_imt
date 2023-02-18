import os
from typing import Any, Dict, Tuple

import cv2
import gym
import numpy as np

from ae.autoencoder import load_ae
from ae.preprocessing import *


class AutoencoderWrapper(gym.Wrapper):
    """
    Wrapper to encode input image using pre-trained AutoEncoder

    :param env: Gym environment
    :param ae_path: absolute path to the pretrained AutoEncoder
    """

    def __init__(self, env: gym.Env, ae_path: str = "/home/rom1/Documents/ia_racing_imt/rl-baselines3-zoo/aae-train-donkeycar/auto-encoder-agent/ae-32_400.pkl",
                 filter=True,degradation=True):
        super().__init__(env)
        self.autoencoder = load_ae(ae_path)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.autoencoder.z_size,),
            dtype=np.float32,
        )
        self.filter = filter
        self.degradation = degradation

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        raw_img = obs[:, :, ::-1]
        # Convert to BGR
        if self.degradation :
            raw_img=degradation(raw_img)
        if self.filter :
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            filtered = final_filter(raw_img)
            filtered = cv2.cvtColor(filtered , cv2.COLOR_GRAY2RGB)
            encoded_image = self.autoencoder.encode_from_raw_image(filtered)
        else :
            encoded_image = self.autoencoder.encode_from_raw_image(raw_img)
        new_obs = np.concatenate([encoded_image.flatten()])
        return new_obs.flatten()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, infos = self.env.step(action)
        # Encode with the pre-trained AutoEncoder
        raw_img = obs[:, :, ::-1]
        # Convert to BGR
        if self.degradation :
            raw_img=degradation(raw_img)
        if self.filter :
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            filtered = final_filter(raw_img)
            filtered = cv2.cvtColor(filtered , cv2.COLOR_GRAY2RGB)
            encoded_image = self.autoencoder.encode_from_raw_image(filtered)
        else :
            encoded_image = self.autoencoder.encode_from_raw_image(raw_img)
        # reconstructed_image = self.autoencoder.decode(encoded_image)[0]
        # cv2.imshow("Original", obs[:, :, ::-1])
        # cv2.imshow("Reconstruction", reconstructed_image)
        # # stop if escape is pressed
        # k = cv2.waitKey(0) & 0xFF
        # if k == 27:
        #     pass
        speed = infos["speed"]
        new_obs = np.concatenate([encoded_image.flatten()])

        return new_obs.flatten(), reward, done, infos
