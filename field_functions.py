import random


def circular(_x, _y):
    return -_y / 100, _x / 100


def constant(_x, _y):
    return 1, 1


def potential1(_x, _y):
    return 0, -_x / 50


def blood(_x, _y):
    xvel = 0 + random.uniform(-1, 1)
    yvel = 3 + random.uniform(-1, 1)
    return xvel, yvel
