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
    yvel = (1 / 37) * _x + _yvel - 1
    print((_y, yvel))
    return xvel, yvel


def stationary(_x, _y, _xvel, _yvel, _xsd, _ysd):
    return 0, 0


def uniform(_x, _y, _xvel, _yvel, _xsd, _ysd):
    xvel = random.uniform(_xvel - _xsd, _xvel + _xsd)
    yvel = random.uniform(_yvel - _ysd, _yvel + _ysd)
    return xvel, yvel


def outliers(_x, _y, _xvel, _yvel, _xsd, _ysd):
    if random.random() >= 0.1:
        xvel = _xvel
        yvel = _yvel
    else:
        xvel = _xvel + _xsd
        yvel = _yvel + _ysd
    return xvel, yvel


def outliers3(_x, _y, _xvel, _yvel, _xsd, _ysd):
    if random.random() >= _xsd:
        xvel = _xvel
        yvel = _yvel
    else:
        xvel = _xvel + _ysd
        yvel = _yvel + _ysd
    return xvel, yvel


def outliers2(_x, _y, _xvel, _yvel, _xsd, _ysd):
    if random.random() >= 0.95:
        xvel = _xvel + _xsd
        yvel = _yvel + _ysd
    elif random.random() <= 0.05:
        xvel = _xvel - _xsd
        yvel = _yvel - _ysd
    else:
        xvel = _xvel
        yvel = _yvel
    return xvel, yvel
