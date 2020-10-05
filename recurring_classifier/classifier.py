import numpy as np

from recurring_classifier.model import get_multilayer_model, get_final_layer_models, get_y_value_map


class Classifier:
    def __init__(self):
        self.multilayer_classifier = get_multilayer_model()
        self.final_layer_models = get_final_layer_models()
        self.value_dict = get_y_value_map()

        self.multi_classes = ("prohibitory", "danger", "direction", "release", "red_surface", "yield", "priority")
        self.final_classes = ("prohibitory", "danger", "direction", "release", "red_surface")

    def predict(self, images):
        labels = self.predict_images(images)
        labels = np.array(labels)
        return labels[labels != -1]

    def predict_images(self, images):
        multilayer_y = self.multilayer_classifier.predict(images)
        images = [[images[img_type][i:i+1] for img_type in range(0, len(images))] for i in range(0, len(images[0]))]
        return [self.predict_class(images[i], multilayer_y[i]) for i in range(0, len(images))]

    def predict_image(self, image):
        image = [np.array([img]) for img in image]
        multilayer_y = self.predict_multilayer_class(image)
        return self.predict_class(image, multilayer_y)

    def predict_multilayer_class(self, image):
        return self.multilayer_classifier.predict(image)[0]

    def predict_class(self, image, multilayer_y):
        groups = multilayer_y.argsort()[::-1][:3]
        for group in groups:
            target = self.multi_classes[group]
            prediction = self.predict_class_by_target(image, target)
            max_pred_idx = np.argmax(prediction)
            if max_pred_idx not in self.value_dict[target]:
                if prediction[max_pred_idx] > multilayer_y[group]:
                    continue
                pred_value = prediction.argsort()[::-1][1]
                return self.value_dict[target][pred_value]
            return self.value_dict[target][max_pred_idx]
        return -1

    def predict_class_by_target(self, image, target):
        if target not in self.final_classes:
            return self.value_dict[target][0]
        model = self.final_layer_models[target]
        return model.predict(image)[0]
