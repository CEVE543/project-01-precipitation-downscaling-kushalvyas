import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(x,y):
    return peak_signal_noise_ratio(y, x, data_range=1.0)


