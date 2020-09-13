import sys
from pathlib import Path


mode = sys.argv[1]
assert mode == 'server' or mode == 'local'

if mode == 'server':
    import server

    server.run()
else:
    from utils import load_image, time
    from trafficsigndetector.traffic_sign_detector import TrafficSignDetector

    image = load_image('{root}/assets/images/{image}.png'.format(root=Path(__file__).parent, image='testimage'))
    detector = TrafficSignDetector()
    time.measure(lambda: detector.detect(image), 'whole process')
