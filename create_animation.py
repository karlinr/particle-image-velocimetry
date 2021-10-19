import numpy as np
import random
import tifffile as tf
import time
import field_functions
import matplotlib.pyplot as plt

# Animation variables
animation_width = 352
animation_height = 352
animation_frames = 10
particles = 1000
particle_size = 9


# Returns a gaussian
# https://mathworld.wolfram.com/GaussianFunction.html
def circular_gaussian(_x, _y, _mean_x, _mean_y, _sd):
    sd2 = 2 * _sd**2
    return (1 / (np.pi * sd2)) * np.exp(-((_x-_mean_x)**2 + (_y - _mean_y)**2) / (sd2)) * 255


class Particle:
    def __init__(self, _x, _y, _function, _brightness):
        # Assign particle variables
        self.function = _function
        self.brightness = _brightness
        self.x = _x
        self.y = _y
        self.velocity_x, self.velocity_y = self.function(self.x, self.y)

    def step(self):
        # Advance a frame
        self.velocity_x, self.velocity_y = self.function(self.x, self.y)
        self.x += self.velocity_x
        self.y += self.velocity_y

        if self.x >= animation_width:
            self.x -= animation_width
        elif self.x < 0:
            self.x += animation_width
        if self.y >= animation_height:
            self.y -= animation_height
        elif self.y < 0:
            self.y += animation_height


def make_animation(_function, _name):
    start = time.time()
    print(f"Creating animation of type {_name}...", end = "")

    # Setup particles
    a = np.empty(particles + 1, dtype=object)
    for i in range(particles):
        _x = random.randint(0, animation_width - 1)
        _y = random.randint(0, animation_height - 1)
        a[i] = Particle(_x, _y, _function, random.uniform(0, 1))

    # Set up video array
    video_array = np.zeros((animation_frames, animation_width, animation_height), dtype=np.float64)

    # Simulate particles
    for t in range(animation_frames):
        image_array = np.zeros((animation_width, animation_height), dtype=np.float64)
        for i in range(particles):
            a[i].step()
            for _x in range(max([0, int(a[i].x - 4)]), min([animation_width, int(a[i].x + 4)])):
                for _y in range(max([0, int(a[i].y - 4)]), min([animation_height, int(a[i].y + 4)])):
                    image_array[_x, _y] += a[i].brightness * circular_gaussian(_x, _y, a[i].x, a[i].y, 2)
        # Save current frame to video
        image_array = np.minimum(image_array, np.full(image_array.shape, 255))
        video_array[t] = image_array

    # Save to disk
    # noinspection PyTypeChecker
    tf.imwrite(f"data/animations/animation_{_name}.tif", video_array.astype(np.int8), compression = "zlib")

    plot_field(_function, _name, animation_width, animation_height, 32)

    # Measure execution time
    end = time.time()
    print(f" completed in {end - start:.2f} seconds")


def plot_field(_function, _name, _width, _height, _window):
    x, y = np.mgrid[int(_window / 2):int(_width):_window, int(_window / 2): int(_height):_window]
    u,v = _function(x,y)
    mag = np.sqrt(pow(u, 2) + pow(v, 2))
    plt.figure()
    plt.quiver(x, y, u, v, mag, cmap = "viridis")
    plt.colorbar()
    plt.xlim(0, animation_width)
    plt.ylim(0, animation_height)
    plt.savefig(f"data/fields/true/{_name}.png")


# Make videos
make_animation(field_functions.circular, "circular")
make_animation(field_functions.constant, "constant")
make_animation(field_functions.potential1, "potential 1")
