import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import keras
from keras import layers
from dataclasses import dataclass



def load_model(ts_dims: int , window_size: int):
    model = keras.Sequential(
            [
                layers.Input(shape=(window_size,  ts_dims)),
                layers.Conv1D(
                    filters=32,
                    kernel_size=7,
                    padding="same",
                    strides=2,
                    activation="relu",
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1D(
                    filters=16,
                    kernel_size=7,
                    padding="same",
                    strides=2,
                    activation="relu",
                ),
                layers.Conv1DTranspose(
                    filters=16,
                    kernel_size=7,
                    padding="same",
                    strides=2,
                    activation="relu",
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1DTranspose(
                    filters=32,
                    kernel_size=7,
                    padding="same",
                    strides=2,
                    activation="relu",
                ),
                layers.Conv1DTranspose(
                    filters= ts_dims, kernel_size=7, padding="same"
                ),
            ]
        )
    
    return model
