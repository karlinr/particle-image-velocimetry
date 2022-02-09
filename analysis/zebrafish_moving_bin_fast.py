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

# number_of_bins = [10, 15, 20, 30, 40, 50]
number_of_bins = range(10, 50)
total_points = 200

files = os.listdir("../data/zebrafish/phase/")

piv = PIV("", 24, 24, 24, 0.6, "5pointgaussian", False)
piv.add_video(["../data/zebrafish/phase/" + str(f) for f in files])
piv.set_coordinate(201, 240)
# piv.get_spaced_coordinates()
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
# piv.plot_flow_field()


for n in number_of_bins:
    ph = []
    vs = []
    std = []
    bins = np.linspace(0, 2 * np.pi, n)

    for binoffset in np.linspace(0, 2 * np.pi / n, total_points // n):
        phases = np.mod([float(os.path.splitext(filename)[0]) + binoffset for filename in files], 2 * np.pi)
        indices = np.digitize(phases, bins)

        for i, b in enumerate(bins):
            argstopiv = np.arange(len(files))[indices == i + 1]
            if argstopiv.shape[0] > 0:
                piv.resample_specific(argstopiv, intensity_array = True)
                piv.get_correlation_averaged_velocity_field()
                # piv.plot_flow_field(f"../analysis/visualisations/02022022/{np.mod(b - binoffset, 2 * np.pi)}.png")
                ph.append(np.mod(b - binoffset, 2 * np.pi))
                vs.append(piv.y_velocity_averaged()[0, 0])
                vs_temp = []
                for i2 in range(50):
                    piv.resample_from_array(argstopiv)
                    piv.get_correlation_averaged_velocity_field()
                    vs_temp.append(piv.y_velocity_averaged().flatten()[0])
                std.append(np.std(vs_temp, ddof = 1))

    ind = np.argsort(ph)
    print(ind)
    ph = np.array(ph)
    vs = np.array(vs)
    std = np.array(std)
    plt.figure(figsize = (8, 8))
    plt.title(f"Displacement for a given phase")
    plt.plot(ph[ind], vs[ind], label = f"Number of bins: {n}", lw = 1, c = "blue")
    plt.fill_between(ph[ind], vs[ind] - std[ind], vs[ind] + std[ind], alpha = 0.2, color = "blue")
    plt.xlabel("Phase (Rads)")
    plt.ylabel("Displacement (px)")
    plt.legend(loc = "upper left")
    # plt.ylim(-10, 30)
    plt.savefig(f"../analysis/visualisations/02022022/optimal_bin_moving/bins_{n}.png")
    plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
