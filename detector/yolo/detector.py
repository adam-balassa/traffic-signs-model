import numpy as np

from detector.non_max_suppression import non_max_suppression
from detector.yolo.images import preprocess_images
from detector.yolo.load import load_model
from detector.yolo.model import get_full_model


class ObjectDetector:
    def __init__(self):
        self.model = get_full_model()
        load_model(self.model)

    def predict(self, image):
        return self.model.predict(np.array([image]))[0]

    def predict_multiple(self, images):
        preprocessed_images = preprocess_images(images)
        predictions = self.model.predict(preprocessed_images)
        mat = np.insert(np.zeros((24, 24, 3)), 1, np.expand_dims(np.full((24, 24), np.arange(24)), axis=-3), axis=-1)
        mat = np.insert(mat, 2,
                        np.expand_dims((np.ones((24, 24)) * np.expand_dims(np.arange(24), -1)).astype('int'), axis=-3),
                        axis=-1)
        normalized = np.array(predictions + np.expand_dims(mat, axis=0))
        print(normalized.astype('float16'))
        normalized[..., 1], normalized[..., 3] = predictions[..., 1] - predictions[..., 3] / 2, predictions[..., 1] + predictions[..., 3] / 2
        normalized[..., 2], normalized[..., 4] = predictions[..., 2] - predictions[..., 4] / 2, predictions[..., 2] + predictions[..., 4] / 2
        normalized = np.reshape(normalized, (len(images), 24 * 24, 5))
        print(normalized[0][500])
        shapes = np.array([image.shape[0] for image in images])
        normalized[..., 1:] *= np.expand_dims(np.expand_dims(shapes / 24, axis=-1), axis=-1)
        return [non_max_suppression(prediction) for prediction in predictions]
