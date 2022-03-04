import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt

vel_x_true = 5.5
vel_x_measured = []
vel_x_std = []
std_errs_to_mean = []

for video in os.listdir("../../data/simulated/constant_for_presentation/"):
    piv = PIV(f"", 24, 24, 1, 0, "5pointgaussian", False)
    piv.add_video(f"../../data/simulated/constant_for_presentation/{video}")
    piv.set_coordinate(36, 36)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vel_x_measured.append(piv.x_velocity_averaged().flatten()[0])
    vel_x_tmp = []
    for _ in range(100):
        piv.resample()
        piv.get_correlation_averaged_velocity_field()
        vel_x_tmp.append(piv.x_velocity_averaged().flatten()[0])
    vel_x_std.append(np.std(vel_x_tmp))
print(np.mean(vel_x_measured))
std_errs_to_mean = (vel_x_measured - np.mean(vel_x_measured)) / vel_x_std
plt.hist(std_errs_to_mean, bins = 50)
plt.show()
print(np.std(std_errs_to_mean))
print(np.mean(std_errs_to_mean))
