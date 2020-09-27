from pathlib import Path

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

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
    objects, labels = time.measure(lambda: detector.detect(image), 'the whole process')
    return HttpResponse(json.convert(objects, labels), content_type="application/json")
