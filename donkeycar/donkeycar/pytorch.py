"""

pytorch.py

Methods to create, use, save and load pilots. Pilots contain the highlevel
logic used to determine the angle and throttle of a vehicle. Pilots can
include one or more models to help direct the vehicles motion.

"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Sequence, Callable
from logging import getLogger


import donkeycar as dk
from donkeycar.utils import normalize_image, linear_bin
from donkeycar.pipeline.types import TubRecord
from donkeycar.parts.interpreter import Interpreter, KerasInterpreter

from donkeycar.parts.autoencoder import load_ae
from sb3_contrib import TQC

ONE_BYTE_SCALE = 1.0 / 255.0

# type of x
XY = Union[float, np.ndarray, Tuple[Union[float, np.ndarray], ...]]

logger = getLogger(__name__)

class CombinedModel():
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def forward(self, x,throttle_history, **kwargs):
        x1 = np.concatenate((self.model1.encode_from_raw_image(x).flatten(),throttle_history))
        x2 = self.model2.predict(x1, deterministic=True)
        return x2



class PytorchReinforcment():
    """
    Base class for Pytorch models that will provide steering and throttle to
    guide a car.
    Possibility to add an autoencoder.
    """

     def __init__(self, throttle_history_dim):
        self.ae = None,
        self.drive = None,
        self.model = None,
        self.throttle_history_dim = throttle_history_dim
        logger.info(f'Created {self}')

    def load_ae(self, path: str) -> None:
        logger.info(f'Loading autoencoder {path}'+ r'\ae.pkl')
        self.ae = load_ae(path+r"\\ae.pkl")

    def load_drive(self, path: str) -> None:
        logger.info(f'Loading drive_model {path}'+r'\drive.zip')
        self.drive = TQC.load(path+r"\\drive.zip")

    def create_model(self) -> None:
        logger.info(f'Creating Model using Autoencoder and Drive models')
        self.model = CombinedModel(self.ae,self.drive)

    def load(self,path:str):
        self.load_ae(path)
        self.load_drive(path)
        print("drive model: ",self.drive)
        self.create_model()


    def shutdown(self) -> None:
        pass

    def compile(self) -> None:
        pass



    def run(self, img_arr: np.ndarray, throttle_history) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """
        Donkeycar parts interface to run the part in the loop.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data to be used in the
                            pilot, like IMU array for the IMU model or a
                            state vector in the Behavioural model
        :return:            tuple of (angle, throttle)
        """
        
        logger.info('model order', order)
        if throttle_history is None:
            throttle_history = [0]*self.throttle_history_dim
        order = self.model.forward(img_arr,throttle_history)[0]
        throttle_history = np.insert(throttle_history,0,order[1])[:-1]
        return order, throttle_history


    def __str__(self) -> str:
        """ For printing model initialisation """
        return type(self).__name__
