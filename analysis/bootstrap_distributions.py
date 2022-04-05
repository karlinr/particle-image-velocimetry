import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy import signal
import random



def Gaussian(x, sigma, mean, a):
    return a * (1 / (sigma * (2 * np.pi) ** (0.5))) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def Uniform(x, min, max, a):
    return np.where((x > min) & (x < max), a, 0)


xs = np.linspace(-5, 15, 6000)
gauss = np.zeros(xs.shape)
uni = Uniform(xs, -1, 1, 1)

dist = np.random.uniform(3, 7, 78)

g = Gaussian(xs, 2.1, 5, 1)
plt.plot(xs, g / np.max(g))

for mean in dist:
    g = Gaussian(xs, 2.1, mean, 1)
    gauss += g

plt.plot(xs, gauss / np.max(gauss))
plt.show()

print(kurtosis(gauss))

piv = PIV(f"", 56, 8, 1, 0, "9pointgaussian", False, True)
piv.add_video(f"../data/simulated/constant/0.tif")
piv.set_coordinate(36, 36)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
tmpx = []
tmpy = []
for _ in range(100000):
    piv.resample()
    piv.get_correlation_averaged_velocity_field()
    tmpx.append(piv.x_velocity_averaged().flatten()[0])
    tmpy.append(piv.y_velocity_averaged().flatten()[0])
"""plt.figure(figsize = (8, 8))
plt.hist2d(tmpx, tmpy, bins = 500)
plt.show()"""
plt.hist(tmpx, bins = 500)
plt.xlim(np.mean(tmpx) - 2, np.mean(tmpx) + 2)
plt.show()
print(np.std(tmpx))

piv = PIV(f"", 56, 8, 1, 0, "9pointgaussian", False, True)
piv.add_video(f"../data/simulated/uniform/0.tif")
piv.set_coordinate(36, 36)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
tmpx = []
tmpy = []
for _ in range(100000):
    # piv.resample(piv.video.shape[0] - 1)
    piv.resample()
    piv.get_correlation_averaged_velocity_field()
    tmpx.append(piv.x_velocity_averaged().flatten()[0])
    tmpy.append(piv.y_velocity_averaged().flatten()[0])
"""plt.figure(figsize = (8, 8))
plt.hist2d(tmpx, tmpy, bins = 500)
plt.show()"""
plt.hist(tmpx, bins = 500)
plt.xlim(np.mean(tmpx) - 2, np.mean(tmpx) + 2)
plt.show()
"""plt.hist(tmpy, bins = 500)
plt.xlim(3, 7)
plt.show()"""

print(np.std(tmpx))
