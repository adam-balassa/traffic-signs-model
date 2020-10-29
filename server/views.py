from pathlib import Path

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np

from server import json
from trafficsigndetector.traffic_sign_detector import TrafficSignDetector
from utils import load_image, time


detector = TrafficSignDetector()


@csrf_exempt
def detect(request):
    body = json.parse(request)
    path = body["path"]
    print(path)
    image = load_image(path)
    objects, labels = time.measure(lambda: detector.detect_multiple(np.asarray([image])), 'the whole process')
    return HttpResponse(json.convert(objects[0], labels[0]), content_type="application/json")
