from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os
import time

start_time = time.time()

# Uses a moving bin to plot correlation averaged velocities

# MPL
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

number_of_bins = [15]
total_points = 100

files = os.listdir("../data/zebrafish/phase/")

piv = PIV("", 24, 24, 24, 0.6, "5pointgaussian", False)
piv.add_video(["../data/zebrafish/phase/" + str(f) for f in files])
piv.set_coordinate(201, 240)
piv.get_correlation_matrices()

for n in number_of_bins:
    ph = []
    vs = []
    std = []

    for binoffset in np.linspace(0, 2 * np.pi / n, total_points // n):

        phases = np.mod([float(os.path.splitext(filename)[0]) + binoffset for filename in files], 2 * np.pi)
        bins = np.linspace(0, 2 * np.pi, n)
        indices = np.digitize(phases, bins)

        for i, b in enumerate(bins):
            argstopiv = np.arange(len(files))[indices == i + 1]
            if argstopiv.shape[0] > 0:
                piv.resample_specific(argstopiv)
                piv.get_correlation_averaged_velocity_field()
                ph.append(np.mod(b - binoffset, 2 * np.pi))
                vs.append(piv.x_velocity_averaged()[0, 0])
                vs_temp = []
                for i2 in range(50):
                    piv.resample_from_array(argstopiv)
                    piv.get_correlation_averaged_velocity_field()
                    vs_temp.append(piv.x_velocity_averaged().flatten()[0])
                std.append(np.std(vs_temp, ddof = 1))

    plt.figure(figsize = (8, 8))
    ind = np.argsort(ph)
    ph = np.array(ph)
    vs = np.array(vs)
    std = np.array(std)
    plt.title(f"Number of Bins: {n}")
    plt.fill_between(ph[ind], vs[ind] - std[ind], vs[ind] + std[ind], alpha = 0.2, color = "black")
    plt.plot(ph[ind], vs[ind], c = "black")
    plt.xlabel("Phase (Rads)")
    plt.ylabel("Displacement (px)")
    plt.ylim(-10, 30)
    plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
