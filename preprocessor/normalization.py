import numpy as np


def mean_norm(imgs):
    means = np.mean(imgs, axis=(1, 2, 3))
    normalized = np.array([img for img in imgs])
    for i in range(0, len(means)):
        normalized[i] -= means[i]
    return normalized


def std_norm(imgs):
    stds = np.std(imgs, axis=(1, 2, 3))
    normalized = np.array([img for img in imgs])
    for i in range(0, len(stds)):
        normalized[i] /= stds[i]
    return normalized


def min_max_norm(imgs):
    low = np.min(imgs, axis=(1, 2, 3))
    high = np.max(imgs, axis=(1, 2, 3))
    normalized = np.array([img for img in imgs])
    for i in range(0, len(imgs)):
        normalized[i] = (normalized[i] - low[i]) / (high[i] - low[i])
    return normalized * 2 - 1


def normalize(imgs):
    imgs = imgs.astype(np.float32)
    imgs = mean_norm(imgs)
    imgs = std_norm(imgs)
    imgs = min_max_norm(imgs)
    return imgs
