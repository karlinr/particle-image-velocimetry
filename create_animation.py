import numpy as np
import random
import tifffile as tf
import time
import field_functions

import matplotlib.pyplot as plt

# Animation variables
animation_width = 352
animation_height = 352
animation_frames = 60
particles = 500
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
        self.set_velocity()

    def set_velocity(self):
        self.velocity_x, self.velocity_y = self.function(self.x, self.y)

    def step(self):
        # Advance a frame
        self.set_velocity()
        self.x += self.velocity_x
        self.y += self.velocity_y

        if self.x >= animation_width:
            self.x -= animation_width
        if self.x < 0:
            self.x += animation_width
        if self.y >= animation_height:
            self.y -= animation_height
        if self.y < 0:
            self.y += animation_height

    def get_position_x(self):
        # Returns particle x position
        return self.x

    def get_position_y(self):
        # Returns particle y position
        return self.y

    def get_brightness(self):
        # Returns particle brightness
        return self.brightness


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
            # Should use non-integer position variables *fix this*
            a[i].step()
            for _x in range(max([0, int(a[i].get_position_x() - 4)]), min([animation_width, int(a[i].get_position_x() + 4)])):
                for _y in range(max([0, int(a[i].get_position_y() - 4)]), min([animation_height, int(a[i].get_position_y() + 4)])):
                    image_array[_x, _y] += a[i].get_brightness() * circular_gaussian(_x, _y, a[i].get_position_x(), a[i].get_position_y(), 2)
        # Save current frame to video
        image_array = np.minimum(image_array, np.full(image_array.shape, 255))
        video_array[t] = image_array

    # Save to disk
    # noinspection PyTypeChecker
    tf.imwrite(f"data/animations/animation_{_name}.tif", video_array.astype(np.int8), compression = "zlib")

    # Measure execution time
    end = time.time()
    print(f" completed in {end - start:.2f} seconds")


# Make videos
make_animation(field_functions.circular, "circular")
make_animation(field_functions.constant, "constant")
make_animation(field_functions.potential1, "potential 1")
make_animation(field_functions.potential2, "potential 2")