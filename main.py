import sys
from pathlib import Path
import numpy as np

from server.server import run

mode = sys.argv[1]
assert mode == 'runserver' or mode == 'local'

if mode == 'runserver':
    run()
else:
    from utils import load_image, time
    from trafficsigndetector.traffic_sign_detector import TrafficSignDetector

    image = load_image('{root}/assets/images/{image}.png'.format(root=Path(__file__).parent, image='testimage'))
    detector = TrafficSignDetector()
    time.measure(lambda: detector.detect_multiple(np.asarray([image for _ in range(0, 2)])), 'whole process')
    #time.measure(lambda: detector.detect(image), 'whole process')
