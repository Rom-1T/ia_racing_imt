import os
from typing import Any, Dict, Tuple
import time

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
                 filter="edge",epaisseur = 1,degrade = False, log = True):
        super().__init__(env)
        self.init_time = time.time()
        #self.autoencoder = load_ae("/home/rom1/Documents/ia_racing_imt/rl-baselines3-zoo/aae-train-donkeycar/auto-encoder-agent/models/ae-32_cam2_edge2.pkl")
        self.autoencoder = load_ae(ae_path)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.autoencoder.z_size,),
            dtype=np.float32,
        )
        self.filter = filter
        self.degradation = degrade
        self.log = log
        self.epaisseur = epaisseur
        name = "data_test"
        print("AE Path : ", ae_path)
        print("degrade : ", self.degradation)
        print("Filter : ", self.filter)
        print("Epaisseur :  ", self.epaisseur)
        print("LOG : ", self.log)
        if self.log :
            self.nbr_img = 0
            i = 0
            directory_path = f"log_img/log_{name}_{i}"
            while os.path.isdir(directory_path):
                i += 1
                directory_path = f"log_img/log_{name}_{i}"
                print("new dir")
            os.makedirs(directory_path)
            self.log_dir = directory_path

    def reset(self) -> np.ndarray:
        self.init_time = time.time()
        obs = self.env.reset()
        raw_img = obs[:, :, ::-1]
        # Convert to BGR
        if self.degradation :
            raw_img=degradation(raw_img)
        if self.filter == "final_filter" :
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            filtered = final_filter(raw_img)
            filtered = cv2.cvtColor(filtered , cv2.COLOR_GRAY2RGB)
            encoded_image = self.autoencoder.encode_from_raw_image(filtered)
        elif self.filter == "edge":
            filtered = processing_line_v2(raw_img,mode="edge",reel=False,epaisseur=self.epaisseur)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            encoded_image = self.autoencoder.encode_from_raw_image(filtered)
        elif self.filter == "lines" :
            filtered = processing_line_v2(raw_img, mode="lines", reel=False, epaisseur=self.epaisseur)
            #filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
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
        if self.log :
            self.nbr_img += 1
            img_to_save = f"/img_{self.nbr_img}.jpg"
            path = self.log_dir + img_to_save
            cv2.imwrite(path, raw_img)
        if self.degradation :
            raw_img=degradation(raw_img)
        if self.filter == "final_filter" :
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            filtered = final_filter(raw_img)
            filtered = cv2.cvtColor(filtered , cv2.COLOR_GRAY2RGB)
            encoded_image = self.autoencoder.encode_from_raw_image(filtered)
        elif self.filter == "edge":
            filtered = processing_line_v2(raw_img,mode="edge",reel=False,epaisseur=self.epaisseur)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            encoded_image = self.autoencoder.encode_from_raw_image(filtered)
            if self.log:
                edge_to_save = f"/img_edge_{self.nbr_img}.jpg"
                line_to_save = f"/img_lines_{self.nbr_img}.jpg"
                cv2.imwrite(self.log_dir + edge_to_save, filtered)
                cv2.imwrite(self.log_dir + line_to_save, processing_line_v2(raw_img, mode="lines", reel=False, epaisseur=self.epaisseur))
        elif self.filter == "lines" :
            filtered = processing_line_v2(raw_img, mode="lines", reel=False, epaisseur=self.epaisseur)
            #filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            encoded_image = self.autoencoder.encode_from_raw_image(filtered)
            if self.log:
                edge = processing_line_v2(raw_img, mode="edge", reel=False, epaisseur=self.epaisseur)
                edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
                edge_to_save = f"/img_edge_{self.nbr_img}.jpg"
                line_to_save = f"/img_lines_{self.nbr_img}.jpg"
                cv2.imwrite(self.log_dir + edge_to_save, edge)
                cv2.imwrite(self.log_dir + line_to_save,filtered )
        else :
            encoded_image = self.autoencoder.encode_from_raw_image(raw_img)
        if self.log :
            img_to_save = f"/img_ae_{self.nbr_img}.jpg"
            path = self.log_dir + img_to_save
            cv2.imwrite(path, self.autoencoder.decode(encoded_image)[0])
        # reconstructed_image = self.autoencoder.decode(encoded_image)[0]
        # cv2.imshow("Original", obs[:, :, ::-1])
        # cv2.imshow("Filtered", filtered)
        # cv2.imshow("Reconstruction", reconstructed_image)
        # # stop if escape is pressed
        # k = cv2.waitKey(0) & 0xFF
        # if k == 27:
        #     pass
        speed = infos["speed"]
        new_obs = np.concatenate([encoded_image.flatten()])

        return new_obs.flatten(), reward, done, infos
