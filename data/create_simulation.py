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
    return np.exp(-((_xdr ** 2 / (2 * _sdx ** 2)) + (_ydr ** 2 / (2 * _sdy ** 2))))


class Particle:
    def __init__(self, _xvel, _yvel, _xsd, _ysd, _function, _theta):
        # Assign particle variables
        self.function = _function
        self.brightness = 8 + np.random.normal(0, 1)
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
        self.brightness = 8 + np.random.normal(0, 1)
        self.x += self.velocity_x
        self.y += self.velocity_y
        #self.theta += random.gauss(0, (2 * np.pi) / 18)

    def randomise_position(self):
        self.velocity_x, self.velocity_y = self.function(self.x, self.y, self.xvel, self.yvel, self.xsd, self.ysd)
        self.x = random.uniform(-particle_size - (simulation_width - animation_width), simulation_width + particle_size)
        self.y = random.uniform(-particle_size - (simulation_height - animation_height), simulation_height + particle_size)


def make_animation(_function, _name, _xvel, _yvel, _xsd, _ysd, num):
    start = time.time()
    print(f"Creating {num} animations of name {_name} with function {_function.__name__}...", end = "")
    if not os.path.exists(f"simulated/{_name}/"):
        os.makedirs(f"simulated/{_name}")
    """if not os.path.exists(f"simulated/outliers_random/{_name}/"):
        os.makedirs(f"simulated/outliers_random/{_name}")"""
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
                image_array = np.add(image_array, a[i].brightness * circular_gaussian(xx, yy, a[i].x, a[i].y, np.random.normal(particle_size, 1), np.random.normal(1.2 * particle_size, 1), a[i].theta))
            image_array = np.add(image_array, np.random.normal(0, 0.5, image_array.shape))
            image_array = np.maximum(np.minimum(image_array, np.full(image_array.shape, 2**16 - 1)), np.full(image_array.shape, 0))
            # Save current frame to video
            video_array[t] = image_array.astype(np.ushort)

        # Save to disk
        tf.imwrite(f"simulated/{_name}/{v}.tif", video_array.astype(np.ushort), compression = "zlib")
        #tf.imwrite(f"simulated/outliers_random/{_name}/{v}.tif", video_array.astype(np.ushort), compression = "zlib")
        """plt.imshow(tf.imread(f"simulated/{_name}/{v}.tif")[0], origin = "lower")
        plt.colorbar()
        plt.show()
        plt.imshow(tf.imread(f"simulated/{_name}/{v}.tif")[1], origin = "lower")
        plt.colorbar()
        plt.show()"""

    # Measure execution time
    end = time.time()
    print(f" completed in {end - start:.2f} seconds")


random.seed()

simulation_width = 96
simulation_height = 96
animation_width = 72
animation_height = 72
animation_frames = 156
particles = 1
particle_size = 14
"""for i in range(1000):
    #vel = random.uniform(0, 2)
    vel = -2
    outlier = random.uniform(0, 4)
    make_animation(field_functions.outliers, f"{vel}_{outlier}", vel, vel, outlier, outlier, 1)"""

make_animation(field_functions.outliers, "no_outliers", -3, -3, 0, 0, 1000)
make_animation(field_functions.outliers, "outliers", -3, -3, 6, 6, 1000)

# Note : 76 is lowest frame count for zebrafish, 194 is max

# Make videos
simulation_width = 96
simulation_height = 96
animation_width = 72
animation_height = 72
animation_frames = 156
particles = 1
particle_size = 14
#make_animation(field_functions.constant, "constant3.5", 0, 3.5, 0, 0, 500)
#make_animation(field_functions.constant, "constant3.25", 0, 3.25, 0, 0, 500)
#make_animation(field_functions.constant, "constant_for_presentation", 5.5, 5.5, 0, 0, 2500)
#make_animation(field_functions.constant, "constant_test_2pass", 5.5, 0, 0, 0, 64)
#make_animation(field_functions.outliers, "outliers_report1", -3, -3, 6, 6, 1)
#make_animation(field_functions.outliers, "outliers_report1none", -3, -3, 0, 0, 1)
#make_animation(field_functions.outliers2, "outliers_report2", -3, -3, 3, 3, 1)
#make_animation(field_functions.constant, "constant_report", 3, 3, 0, 0, 2500)

"""for i, r in enumerate(np.linspace(0, 0.25, 11)):
    make_animation(field_functions.outliers3, f"outliers_report_{i}", -3, -3, r, 6, 30)"""

"""for i in range(12):
    avg_velocity = -3
    outlier_velocity = i
    velocity = (avg_velocity - 0.2 * outlier_velocity) / 0.8
    #print(0.8 * velocity + 0.2 * outlier_velocity)
    print(velocity)
    make_animation(field_functions.outliers, f"outliers_report_{i}", velocity, velocity, outlier_velocity, outlier_velocity, 100)"""

"""make_animation(field_functions.outliers, "outliers_report_12", -6, -6, 12, 12, 100)
make_animation(field_functions.outliers, "outliers_report_11", -6, -6, 11, 11, 100)
make_animation(field_functions.outliers, "outliers_report_10", -6, -6, 10, 10, 100)
make_animation(field_functions.outliers, "outliers_report_9", -6, -6, 9, 9, 100)
make_animation(field_functions.outliers, "outliers_report_8", -6, -6, 8, 8, 100)
make_animation(field_functions.outliers, "outliers_report_7", -6, -6, 7, 7, 100)
make_animation(field_functions.outliers, "outliers_report_6", -6, -6, 6, 6, 100)
make_animation(field_functions.outliers, "outliers_report_5", -6, -6, 5, 5, 100)
make_animation(field_functions.outliers, "outliers_report_4", -6, -6, 4, 4, 100)
make_animation(field_functions.outliers, "outliers_report_3", -6, -6, 3, 3, 100)
make_animation(field_functions.outliers, "outliers_report_2", -6, -6, 2, 2, 100)
make_animation(field_functions.outliers, "outliers_report_1", -6, -6, 1, 1, 100)
make_animation(field_functions.outliers, "outliers_report_0", -6, -6, 0, 0, 100)"""


# Make videos
#simulation_width = 96
#simulation_height = 96
#animation_width = 72
#animation_height = 72
#animation_frames = 156
#particles = 1
#particle_size = 8
#make_animation(field_functions.uniform, "uniform", 5, 5, 2, 2, 1)
#make_animation(field_functions.constant, "constant", 5, 5, 0, 0, 1)

# Make videos
simulation_width = 96
simulation_height = 96
animation_width = 72
animation_height = 72
animation_frames = 156
particles = 0
particle_size = 6
#make_animation(field_functions.uniform, "constant_small", 1, 1, 0, 0, 200)
