import numpy as np
import tifffile as tf
import time
import field_functions
import random
import os


# Returns a gaussian
# https://mathworld.wolfram.com/GaussianFunction.html
def circular_gaussian(_x, _y, _mean_x, _mean_y, _sdx, _sdy, _theta):
    _xd = (_x - _mean_x)
    _yd = (_y - _mean_y)
    _xdr = _xd * np.cos(_theta) - _yd * np.sin(_theta)
    _ydr = _xd * np.sin(_theta) + _yd * np.cos(_theta)
    return (2**16 - 1) * np.exp(-((_xdr ** 2 / (2 * _sdx ** 2)) + (_ydr ** 2 / (2 * _sdy ** 2))))


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

    def randomise_position(self):
        self.x = random.uniform(-particle_size, animation_width + particle_size)
        self.y = random.uniform(-particle_size, animation_height + particle_size)


def make_animation(_function, _name, _xvel, _yvel, _xsd, _ysd, num):
    start = time.time()
    print(f"Creating {num} animations of name {_name} with function {_function}...", end = "")
    if not os.path.exists(f"data/simulated/{_name}/"):
        os.makedirs(f"data/simulated/{_name}")
    for v in range(num):

        # Setup particles
        a = np.empty(particles + 1, dtype=object)
        for i in range(particles):
            _x = random.uniform(-particle_size, animation_width + particle_size)
            _y = random.uniform(-particle_size, animation_height + particle_size)

            a[i] = Particle(_x, _y, _xvel, _yvel, _xsd, _ysd, _function, 0.15 + 0.15 * random.uniform(0, 1), random.uniform(0, 2 * np.pi))

        # Set up video array
        video_array = np.zeros((animation_frames, animation_width, animation_height), dtype=np.int_)
        xx, yy = np.meshgrid(range(animation_width), range(animation_height))

        # Simulate particles
        for t in range(animation_frames):
            image_array = np.zeros((animation_width, animation_height), dtype=np.int_)
            for i in range(particles):
                if t % 2 == 0:
                    a[i].randomise_position()
                a[i].step()
                image_array = np.add(image_array, a[i].brightness * circular_gaussian(xx, yy, a[i].x, a[i].y, particle_size, particle_size, a[i].theta))
            # Save current frame to video
            image_array = np.minimum(image_array, np.full(image_array.shape, 2**16 - 1))
            video_array[t] = image_array

        # Save to disk
        tf.imwrite(f"simulated/{_name}/{v}.tif", video_array.astype(np.ushort), compression = "zlib")

    # Measure execution time
    end = time.time()
    print(f" completed in {end - start:.2f} seconds")


random.seed()

# Note : 76 is lowest frame count for zebrafish, 194 is max

# Make videos
animation_width = 54
animation_height = 54
animation_frames = 90
particles = 1
particle_size = 7
make_animation(field_functions.constant, "constant_vx3-25_vxsd1-0_vy3-25_vysd1-0_f500", 3.25, 3.25, 1, 1, 500)
make_animation(field_functions.constant, "constant_vx0-0_vxsd0-0_vy3-0_vysd0-0_f500", 0, 3, 0, 0, 500)
make_animation(field_functions.constant, "constant_vx3-0_vxsd0-0_vy0-0_vysd0-0_f500", 3, 0, 0, 0, 500)
make_animation(field_functions.constant, "constant_vx3-0_vxsd0-0_vy3-0_vysd0-0_f500", 3, 3, 0, 0, 500)
make_animation(field_functions.constant, "constant_vx3-25_vxsd0-0_vy3-25_vysd0-0_f500", 3.25, 3.25, 0, 0, 500)
make_animation(field_functions.constant, "constant_vx3-5_vxsd0-0_vy3-5_vysd0-0_f500", 3.5, 3.5, 0, 0, 500)
