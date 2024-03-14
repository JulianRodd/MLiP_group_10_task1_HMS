import numpy as np
import pywt


def maddest(d, axis=None):
    """Calculate the Mean Absolute Deviation."""
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    """Denoise a signal using Discrete Wavelet Transform."""
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="per")
