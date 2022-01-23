from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


for filename in os.listdir(f"../data/zebrafish/processed/"):
    piv = PIV(f"../data/zebrafish/processed/{filename}", 24, 24, 16, 0, "5pointgaussian", True)
    xs = []
    piv.set_coordinate(230, 280)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    piv.plot_flow_field()
    for i in range(500):
        piv.resample()
        piv.get_correlation_averaged_velocity_field()
        xs.append(piv.x_velocity_averaged())

    print(np.mean(xs))
    print(np.std(xs))