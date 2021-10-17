import numpy as np
import random
import tifffile as tf
import time

import matplotlib.pyplot as plt

# Animation variables
animation_width = 352
animation_height = 352
animation_frames = 20
particles = 10000
particle_size = 9


# Returns a gaussian
# https://mathworld.wolfram.com/GaussianFunction.html
def circular_gaussian(_x, _y, _mean_x, _mean_y, _sd):
    return (1 / (1*np.pi * _sd**2)) * np.exp(-((_x-_mean_x)**2 + (_y - _mean_y)**2) / (2 * _sd**2)) * 255


class Particle:
    def __init__(self, _x, _y, _particle_type, _brightness):
        # Assign particle variables
        self.particle_type = _particle_type
        self.brightness = _brightness

        if self.particle_type == "circular":
            self.radius = random.uniform(1, np.min([animation_width, animation_height]) / 2)
            self.theta = random.uniform(0, 360)
        else:
            self.x = _x
            self.y = _y
            self.velocity_x = 0
            self.velocity_y = 0
            self.set_velocity()

    def set_velocity(self):
        # Particle motion
        if self.particle_type == "potential":
            self.velocity_x = 0
            self.velocity_y = 1 + self.y / 80
        elif self.particle_type == "constant":
            self.velocity_x = 3
            self.velocity_y = 3
        elif self.particle_type == "random":
            self.velocity_x = random.uniform(-3, 3)
            self.velocity_y = random.uniform(-3, 3)
        elif self.particle_type == "noise":
            self.x = random.randint(0, animation_width - 1)
            self.y = random.randint(0, animation_height - 1)

    def step(self):
        # Advance a frame
        if self.particle_type == "circular":
            # Update angle and position
            self.theta += 2 / self.radius
            self.x = animation_width / 2 + self.radius * np.cos(self.theta)
            self.y = animation_height / 2 + self.radius * np.sin(self.theta)
        else:
            # Update velocity and position
            self.set_velocity()
            self.x += self.velocity_x
            self.y += self.velocity_y

            # Handle out of frame
            if self.x >= animation_width:
                self.x -= animation_width
            if self.x < 0:
                self.x += animation_width
            if self.y >= animation_height:
                self.y -= animation_height
            if self.y < 0:
                self.y += animation_height

    def get_position(self):
        # Returns particle position as a tuple
        return int(self.x), int(self.y)

    def get_brightness(self):
        # Returns particle brightness
        return self.brightness


def make_animation(_particle_type):
    gaussian = np.zeros((particle_size, particle_size))

    for _x in range(particle_size):
        for _y in range(particle_size):
            gaussian[_x, _y] = circular_gaussian(_x, _y, particle_size / 2 - 0.5, particle_size / 2 - 0.5, particle_size / 8)
    plt.imshow(gaussian)
    plt.show()

    start = time.time()
    print(f"Creating animation of type {_particle_type}...", end = "")

    # Setup particles
    a = np.empty(particles + 1, dtype=object)
    for i in range(particles):
        _x = random.randint(0, animation_width - 1)
        _y = random.randint(0, animation_height - 1)
        a[i] = Particle(_x, _y, _particle_type, random.randint(0, 1))

    # Set up video array
    video_array = np.zeros((animation_frames, animation_width, animation_height), dtype=np.float64)

    # Simulate particles
    for t in range(animation_frames):
        image_array = np.zeros((animation_width, animation_height), dtype=np.float64)
        for i in range(particles):
            a[i].step()
            array_size = image_array[a[i].get_position()[0]:a[i].get_position()[0] + particle_size, a[i].get_position()[1]: a[i].get_position()[1] + particle_size].shape
            image_array[a[i].get_position()[0]:a[i].get_position()[0] + particle_size, a[i].get_position()[1]: a[i].get_position()[1] + particle_size] += a[i].get_brightness() * gaussian[0:array_size[0],0:array_size[1]]
        # Save current frame to video
        image_array = np.minimum(image_array, np.full(image_array.shape,255))
        video_array[t] = image_array

    # Save to disk
    # noinspection PyTypeChecker
    print(np.max(video_array))
    tf.imwrite(f"data/animations/animation_{_particle_type}.tif", video_array.astype(np.int8), compression = "zlib")

    # Measure execution time
    end = time.time()
    print(f" completed in {end - start:.2f} seconds")


# Make videos
make_animation("circular")
make_animation("random")
make_animation("constant")
make_animation("potential")
make_animation("noise")
