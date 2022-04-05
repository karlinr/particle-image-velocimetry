import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt

vel_x_true = 5.5
vel_x_measured = []
vel_x_std = []
std_errs_to_mean = []

piv = PIV(f"", 24, 24, 1, 0, "5pointgaussian", False)
piv.add_video(f"../../data/simulated/constant_test_2pass/0.tif")
piv.set_coordinate(36, 36)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
print(piv.y_velocity_averaged())
piv.do_pass()
piv.set(16, 16, 1)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
print(piv.y_velocity_averaged())