import random


def constant(_x, _y, _xvel, _yvel, _xsd, _ysd):
    xvel = random.gauss(_xvel, _xsd)
    yvel = random.gauss(_yvel, _ysd)
    return xvel, yvel


def constant_with_gradient(_x, _y, _xvel, _yvel, _xsd, _ysd):
    if _x > 160:
        xvel = random.gauss(_xvel, _xsd)
        yvel = random.gauss(_yvel, _ysd)
    else:
        xvel = -random.gauss(_xvel, _xsd)
        yvel = -random.gauss(_yvel, _ysd)
    return xvel, yvel


def stationary(_x, _y, _xvel, _yvel, _xsd, _ysd):
    return 0, 0