import numpy as np
import tifffile as tf
import time
import field_functions
import random
import os
import matplotlib.pyplot as plt


# Returns a gaussian
# https://mathworld.wolfram.com/GaussianFunction.html
def circular_gaussian(_x, _y, _mean_x, _mean_y, _sdx, _sdy, _theta):
    _xd = (_x - _mean_x)
    _yd = (_y - _mean_y)
    _xdr = _xd * np.cos(_theta) - _yd * np.sin(_theta)
    _ydr = _xd * np.sin(_theta) + _yd * np.cos(_theta)
    return (2**16 - 1) * np.exp(-((_xdr ** 2 / (2 * _sdx ** 2)) + (_ydr ** 2 / (2 * _sdy ** 2))))


class Particle:
    def __init__(self, _xvel, _yvel, _xsd, _ysd, _function, _theta):
        # Assign particle variables
        self.function = _function
        self.brightness = (0.005 + 0.0005 * random.uniform(0, 1)) / 100
        self.x = 0
        self.y = 0
        self.xvel = _xvel
        self.yvel = _yvel
        self.xsd = _xsd
        self.ysd = _ysd
        self.theta = _theta
        self.velocity_x, self.velocity_y = self.function(self.x, self.y, self.xvel, self.yvel, self.xsd, self.ysd)

        self.randomise_position()

    def step(self):
        # Advance a frame
        self.brightness = (0.005 + 0.0005 * random.uniform(0, 1)) / 100
        self.velocity_x, self.velocity_y = self.function(self.x, self.y, self.xvel, self.yvel, self.xsd, self.ysd)
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.theta += random.gauss(0, np.pi / 20)

    def randomise_position(self):
        self.x = random.uniform(-particle_size - (simulation_width - animation_width), simulation_width + particle_size)
        self.y = random.uniform(-particle_size - (simulation_height - animation_height), simulation_height + particle_size)


def make_animation(_function, _name, _xvel, _yvel, _xsd, _ysd, num):
    start = time.time()
    print(f"Creating {num} animations of name {_name} with function {_function}...", end = "")
    if not os.path.exists(f"simulated/{_name}/"):
        os.makedirs(f"simulated/{_name}")
    for v in range(num):

        # Setup particles
        a = np.empty(particles + 1, dtype=object)
        for i in range(particles):
            _x = random.uniform(-particle_size - (simulation_width - animation_width), simulation_width + particle_size)
            _y = random.uniform(-particle_size - (simulation_height - animation_height), simulation_height + particle_size)

            a[i] = Particle(_xvel, _yvel, _xsd, _ysd, _function, random.uniform(0, 2 * np.pi))

        # Set up video array
        video_array = np.zeros((animation_frames, animation_width, animation_height), dtype = np.ushort)
        xx, yy = np.meshgrid(range(animation_width), range(animation_height))

        # Simulate particles
        for t in range(animation_frames):
            image_array = np.full((animation_width, animation_height), 8, dtype = np.ushort)
            for i in range(particles):
                if t % 2 == 0:
                    a[i].randomise_position()
                a[i].step()
                image_array = np.add(image_array, a[i].brightness * circular_gaussian(xx, yy, a[i].x, a[i].y, particle_size + random.gauss(0.5, 0.2), 2 * particle_size + random.gauss(1, 1), a[i].theta))
            image_array = np.add(image_array, np.random.normal(0, 0.3, image_array.shape))
            image_array = np.maximum(np.minimum(image_array, np.full(image_array.shape, 2**16 - 1)), np.full(image_array.shape, 0))
            # Save current frame to video
            video_array[t] = image_array.astype(np.ushort)
            #plt.imshow(video_array[t], origin = "lower")
            #plt.show()

        # Save to disk
        tf.imwrite(f"simulated/{_name}/{v}.tif", video_array.astype(np.ushort), compression = "zlib")

    # Measure execution time
    end = time.time()
    print(f" completed in {end - start:.2f} seconds")


random.seed()

# Note : 76 is lowest frame count for zebrafish, 194 is max

# Make videos
simulation_width = 96
simulation_height = 96
animation_width = 72
animation_height = 72
animation_frames = 128
particles = 5
particle_size = 9
#make_animation(field_functions.constant, "constant3.5", 0, 3.5, 0, 0, 500)
#make_animation(field_functions.constant, "constant3.25", 0, 3.25, 0, 0, 500)
make_animation(field_functions.constant, "constant_for_presentation", 5.5, 5.5, 0, 0, 500)
