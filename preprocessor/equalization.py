import numpy as np
from skimage import exposure


def histogram_stretching(img):
    p2 = np.percentile(img, 0)
    p98 = np.percentile(img, 85)
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale


def histogram_equalization(img):
    imgEq = exposure.equalize_hist(img)
    return imgEq


def adaptive_histogram_equalization(imgs):
    imgEq = exposure.equalize_adapthist(imgs, clip_limit=0.1)
    return imgEq
