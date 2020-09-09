import cv2


def load_image(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = im.astype('float32')
    im = im / 255.0
    return im
