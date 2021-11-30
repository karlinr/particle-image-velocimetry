import random


def constant(_x, _y, _xvel, _yvel, _xsd, _ysd):
    xvel = random.gauss(_xvel, _xsd)
    yvel = random.gauss(_yvel, _ysd)
    return xvel, yvel