from cv2 import resize
import skimage.feature as feature

from preprocessor import histogram_equalization


def hog(img):
    img = resize(img, (300, 300))
    img = histogram_equalization(img)
    data = feature.hog(img, orientations=8, pixels_per_cell=(12, 12), cells_per_block=(1, 1), multichannel=True)
    return data
