import torch
import numpy as np  

C_PI = 3.14159265358979323846


def RF_path_loss_torch(distance, frequency=77e9):
    c = 3e8  # Speed of light in m/s
    path_loss_db = 20 * torch.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10((4 * np.pi) / c)
    path_loss_percentage = 10 ** (-path_loss_db / 20)
    return path_loss_percentage

def db_to_linear(db):
    return 10**(db/20)

def FSPL(distance, frequency=77e9):
    wavelength = 3e8 / frequency
    return (wavelength / (4 * C_PI * distance)) ** 2
