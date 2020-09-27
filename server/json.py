from datetime import datetime
import json
import numpy as np


def convert(objects, labels):
    result = {'objects': objects, 'classifications': labels}
    return json.dumps(result, default=universal_converter)


def parse(request):
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    return body


def universal_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.__str__()
