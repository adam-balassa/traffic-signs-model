import time


def measure(fn, message):
    start = time.time()
    result = fn()
    end = time.time()
    print(f'{message} took {"{:.2f}".format(end - start)}s')
    return result
