from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# Systematically remove outliers from left and right to see if correlation averaged return to mean

# MPL
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

for filename in os.listdir("../../data/zebrafish/processed/"):
    piv = PIV(f"", 24, 27, 24, 0, "5pointgaussian", False)
    piv.add_video(f"../../data/zebrafish/processed/{filename}")
    piv.set_coordinate(201, 240)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    piv.get_velocity_field()

    mean_vx = piv.x_velocity_averaged().flatten()[0]
    mean_vy = piv.y_velocity_averaged().flatten()[0]
    tmp1 = []
    tmp2 = []
    for i in range(100):
        piv.resample()
        piv.get_correlation_averaged_velocity_field()
        tmp1.append(piv.x_velocity_averaged().flatten()[0])
        tmp2.append(piv.x_velocity_averaged().flatten()[0])
    mean_vx_std = np.std(tmp1)
    mean_vy_std = np.std(tmp2)

    vs_x = piv.x_velocity().flatten()
    vs_y = piv.y_velocity().flatten()

    vx_left = []
    vx_left_std = []

    for _vx in sorted(vs_x):
        arr = np.flatnonzero(vs_x >= _vx)
        piv.resample_specific(arr)
        piv.get_correlation_averaged_velocity_field()
        vx_left.append(piv.x_velocity_averaged().flatten()[0])
        tmp = []
        for i in range(100):
            piv.resample_from_array(arr)
            piv.get_correlation_averaged_velocity_field()
            tmp.append(piv.x_velocity_averaged().flatten()[0])
        vx_left_std.append(np.std(tmp))

    vy_left = []
    vy_left_std = []

    for _vy in sorted(vs_y):
        arr = np.flatnonzero(vs_y >= _vy)
        piv.resample_specific(arr)
        piv.get_correlation_averaged_velocity_field()
        vy_left.append(piv.y_velocity_averaged().flatten()[0])
        tmp = []
        for i in range(100):
            piv.resample_from_array(arr)
            piv.get_correlation_averaged_velocity_field()
            tmp.append(piv.y_velocity_averaged().flatten()[0])
        vy_left_std.append(np.std(tmp))

    plt.plot(range(len(vs_x)), vx_left, color = "blue")
    plt.fill_between(range(len(vs_x)), np.array(vx_left) + np.array(vx_left_std), np.array(vx_left) - np.array(vx_left_std), color = "blue", alpha = 0.1)
    plt.plot(range(len(vs_x)), sorted(vs_x), c = "orange")
    plt.axhline(mean_vx, c = "black")
    plt.axhline(mean_vx + mean_vx_std, ls = ":", c = "black")
    plt.axhline(mean_vx - mean_vx_std, ls = ":", c = "black")
    #plt.savefig(f"../analysis/visualisations/02022022/outliers_compare/x/left/{filename}")
    plt.xlabel("Number Excluded")
    plt.ylabel("y displacement (px)")
    plt.show()

    plt.plot(range(len(vs_y)), vy_left, color = "blue")
    plt.fill_between(range(len(vs_y)), np.array(vy_left) + np.array(vy_left_std), np.array(vy_left) - np.array(vy_left_std), color = "blue", alpha = 0.1)
    plt.plot(range(len(vs_y)), sorted(vs_y), c = "orange")
    plt.axhline(mean_vy, c = "black")
    plt.axhline(mean_vy + mean_vy_std, ls = ":", c = "black")
    plt.axhline(mean_vy - mean_vy_std, ls = ":", c = "black")
    #plt.savefig(f"../analysis/visualisations/02022022/outliers_compare/y/left/{filename}")
    plt.xlabel("Number Excluded")
    plt.ylabel("y displacement (px)")
    plt.show()
