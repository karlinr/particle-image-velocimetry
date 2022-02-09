from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# Systematically remove outliers from left and right to see if correlation averaged return to mean

# MPL
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

for filename in os.listdir("../data/zebrafish/processed/"):
    piv = PIV(f"", 24, 27, 24, 0, "5pointgaussian", False)
    piv.add_video(f"../data/zebrafish/processed/{filename}")
    piv.set_coordinate(201, 240)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    piv.get_velocity_field()

    mean_vel = [piv.x_velocity_averaged().flatten()[0], piv.y_velocity_averaged().flatten()[0]]
    mean_spd = np.sqrt(piv.x_velocity_averaged().flatten()[0]**2 + piv.y_velocity_averaged().flatten()[0]**2)
    temp = []
    for i in range(500):
        piv.resample()
        piv.get_correlation_averaged_velocity_field()
        temp.append(np.sqrt(piv.x_velocity_averaged().flatten()[0]**2 + piv.y_velocity_averaged().flatten()[0]**2))
    mean_spd_std = np.std(temp)

    uppers = []
    uppers_std = []
    vdots = []

    for i in range(piv.x_velocity().shape[0]):
        vdots.append(np.dot([piv.x_velocity().flatten()[i], piv.y_velocity().flatten()[i]], mean_vel) / mean_spd)

    for vdot in sorted(vdots, reverse = True):
        arr = np.flatnonzero(vdots <= vdot)
        piv.resample_specific(arr)
        piv.get_correlation_averaged_velocity_field()
        uppers.append(np.dot([piv.x_velocity_averaged().flatten()[0], piv.y_velocity_averaged().flatten()[0]], mean_vel) / mean_spd)
        temp = []
        for i in range(500):
            piv.resample_from_array(arr)
            piv.get_correlation_averaged_velocity_field()
            temp.append(np.dot([piv.x_velocity_averaged().flatten()[0], piv.y_velocity_averaged().flatten()[0]], mean_vel) / np.sqrt(mean_vel[0]**2 + mean_vel[1]**2))
        uppers_std.append(np.std(temp))

    plt.title(f"{filename}")
    #plt.scatter(range(len(vdots)), uppers, s = 2, c = "blue")
    plt.plot(range(len(vdots)), uppers, c = "blue")
    plt.fill_between(range(len(vdots)), np.array(uppers) + np.array(uppers_std), np.array(uppers) - np.array(uppers_std), color = "blue", alpha = 0.1)
    plt.axhline(mean_spd, c = "black")
    plt.axhline(mean_spd + mean_spd_std, ls = ":", c = "black")
    plt.axhline(mean_spd - mean_spd_std, ls = ":", c = "black")
    plt.plot(range(len(vdots)), sorted(vdots, reverse = True), c = "orange")
    plt.xlabel("Number Excluded")
    plt.ylabel("Displacement in direction of mean vector (px)")
    plt.savefig(f"../analysis/visualisations/02022022/outliers_compare_dot/{filename}")
    plt.show()