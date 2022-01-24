import random


def constant(_x, _y, _xvel, _yvel, _xsd, _ysd):
    xvel = random.gauss(_xvel, _xsd)
    yvel = random.gauss(_yvel, _ysd)
    return xvel, yvel


def contra(_x, _y, _xvel, _yvel, _xsd, _ysd):
    if _y < 40:
        xvel = random.gauss(-_xvel, _xsd)
        yvel = random.gauss(-_yvel, _ysd)
    else:
        xvel = random.gauss(_xvel, _xsd)
        yvel = random.gauss(_yvel, _ysd)
    return xvel, yvel


def gradient(_x, _y, _xvel, _yvel, _xsd, _ysd):
    xvel = random.gauss(_xvel, _xsd)
    yvel = 0.5 * (_yvel + (1/40) * _y * _yvel)
    return xvel, yvel


def stationary(_x, _y, _xvel, _yvel, _xsd, _ysd):
    return 0, 0