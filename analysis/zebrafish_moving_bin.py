from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# MPL
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

number_of_bins = 30

ph = []
vs = []
std = []
np.set_printoptions(threshold = np.inf)

plt.figure(figsize = (8, 8))
for binoffset in np.linspace(0, 2 * np.pi / number_of_bins, 10):
    import time

    start_time = time.time()
    files = os.listdir("../data/zebrafish/unbinned/")
    phases = np.mod([float(os.path.splitext(filename)[0]) + binoffset for filename in files], 2 * np.pi)
    bins = np.linspace(0, 2 * np.pi, number_of_bins)
    indices = np.digitize(phases, bins)

    for i, b in enumerate(bins):
        filestopiv = np.array(files)[indices == i + 1]
        if filestopiv.shape[0] > 0:
            piv = PIV(b, 24, 24, 24, 0.6, "5pointgaussian", False)
            piv.add_video(["../data/zebrafish/unbinned/" + str(f) for f in filestopiv])
            piv.set_coordinate(201, 240)
            piv.get_correlation_matrices()
            piv.get_correlation_averaged_velocity_field()
            ph.append(np.mod(b - binoffset, 2 * np.pi))
            vs.append(piv.x_velocity_averaged()[0, 0])
            vs_temp = []
            for i in range(5):
                piv.resample()
                piv.get_correlation_averaged_velocity_field()
                vs_temp.append(piv.x_velocity_averaged().flatten()[0])
            std.append(np.std(vs_temp, ddof = 1))
    print("--- %s seconds ---" % (time.time() - start_time))

ind = np.argsort(ph)
ph = np.array(ph)
vs = np.array(vs)
std = np.array(std)
print(ph[ind])
print(vs[ind])
print(std[ind])
print(ind)
plt.fill_between(ph[ind], vs[ind] - std[ind], vs[ind] + std[ind], alpha = 0.2, color = "black")
plt.plot(ph[ind], vs[ind], c = "black")
plt.xlabel("Phase (Rads)")
plt.ylabel("Displacement (px)")
plt.show()