import numpy as np
import random
import tifffile as tf
import time
import field_functions
import matplotlib.pyplot as plt
import random

random.seed()

# Animation variables
animation_width = 54
animation_height = 54
animation_frames = 60
particles = 1
particle_size = 6


# Returns a gaussian
# https://mathworld.wolfram.com/GaussianFunction.html
def circular_gaussian(_x, _y, _mean_x, _mean_y, _sdx, _sdy, _theta):
    _xd = (_x - _mean_x)
    _yd = (_y - _mean_y)
    _xdr = _xd * np.cos(_theta) - _yd * np.sin(_theta)
    _ydr = _xd * np.sin(_theta) + _yd * np.cos(_theta)
    return (2**16 - 1) * np.exp(-((_xdr ** 2 / (2 * _sdx ** 2)) + ((_ydr) ** 2 / (2 * _sdy ** 2))))


class Particle:
    def __init__(self, _x, _y, _xvel, _yvel, _xsd, _ysd, _function, _brightness, _theta):
        # Assign particle variables
        self.function = _function
        self.brightness = _brightness
        self.x = _x
        self.y = _y
        self.xvel = _xvel
        self.yvel = _yvel
        self.xsd = _xsd
        self.ysd = _ysd
        self.theta = _theta
        self.theta = 0
        self.velocity_x, self.velocity_y = self.function(self.x, self.y, self.xvel, self.yvel, self.xsd, self.ysd)

    def step(self):
        # Advance a frame
        self.velocity_x, self.velocity_y = self.function(self.x, self.y, self.xvel, self.yvel, self.xsd, self.ysd)
        self.x += self.velocity_x
        self.y += self.velocity_y

        #self.theta += random.gauss(0, (np.pi * 2) / 360)

        """if self.x >= animation_width:
            self.x -= animation_width
        elif self.x < 0:
            self.x += animation_width
        if self.y >= animation_height:
            self.y -= animation_height
        elif self.y < 0:
            self.y += animation_height"""

    def randomise_position(self):
        self.x = random.uniform(-particle_size, animation_width + particle_size)
        self.y = random.uniform(-particle_size, animation_height + particle_size)


def make_animation(_function, _name, _xvel, _yvel, _xsd, _ysd):
    start = time.time()
    print(f"Creating animation of type {_name}...")

    # Setup particles
    a = np.empty(particles + 1, dtype=object)
    for i in range(particles):
        _x = random.uniform(-particle_size, animation_width + particle_size)
        _y = random.uniform(-particle_size, animation_height + particle_size)

        a[i] = Particle(_x, _y, _xvel, _yvel, _xsd, _ysd, _function, 0.15 + 0.15 * random.uniform(0, 1), random.uniform(0, 2 * np.pi))

    # Set up video array
    video_array = np.zeros((animation_frames, animation_width, animation_height), dtype=np.int_)

    xs = []
    vs = []

    # Simulate particles
    for t in range(animation_frames):
        image_array = np.zeros((animation_width, animation_height), dtype=np.int_)
        for i in range(particles):
            if t % 2 == 0:
                a[i].randomise_position()
                xs.append(a[i].x)
                vs.append(a[i].velocity_x)
            a[i].step()
            """for _x in range(max([0, int(a[i].x - particle_size * 3)]),
                            min([animation_width, int(a[i].x + particle_size * 3)])):
                for _y in range(max([0, int(a[i].y - particle_size * 3)]),
                                min([animation_height, int(a[i].y + particle_size * 3)])):"""
            for _x in range(0, animation_width):
                for _y in range(0, animation_height):
                    image_array[_x, _y] += a[i].brightness * circular_gaussian(_x, _y, a[i].x, a[i].y, particle_size, particle_size, a[i].theta)
        # Save current frame to video
        image_array = np.minimum(image_array, np.full(image_array.shape, 2**16 - 1))
        video_array[t] = image_array
        print(f"Completed {t} of {animation_frames} frames")

    plt.hist(xs)
    plt.show()
    plt.hist(vs)
    plt.show()

    # Save to disk
    # noinspection PyTypeChecker
    tf.imwrite(f"data/animations/animation_{_name}.tif", video_array.astype(np.ushort), compression = "zlib")

    #plot_field(_function, _name, animation_width, animation_height, 32, _xvel, _yvel)

    # Measure execution time
    end = time.time()
    print(f" completed in {end - start:.2f} seconds")


def plot_field(_function, _name, _width, _height, _window, _xvel, _yvel):
    x, y = np.mgrid[int(_window / 2):int(_width):_window, int(_window / 2): int(_height):_window]
    u, v = _function(x, y, _xvel, _yvel, 0, 0)
    mag = np.sqrt(pow(u, 2) + pow(v, 2))
    plt.figure()
    plt.quiver(x, y, u, v, mag, cmap = "viridis")
    plt.colorbar()
    plt.xlim(0, animation_width)
    plt.ylim(0, animation_height)
    plt.savefig(f"data/fields/true/{_name}.png")


# Make videos
#make_animation(field_functions.constant, "constant", 6, 0, 0, 0)
make_animation(field_functions.constant_with_gradient, "constant_with_gradient", 3, 0, 0, 0)
#make_animation(field_functions.constant, "stationary", 0, 0, 0, 0)
