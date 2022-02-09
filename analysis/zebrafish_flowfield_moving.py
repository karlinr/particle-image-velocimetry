from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os
import time

start_time = time.time()

# Uses a moving bin to plot correlation averaged velocities

# MPL
plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

number_of_bins = 25
total_points = 200
bins = np.linspace(0, 2 * np.pi, number_of_bins)

files = os.listdir("../data/zebrafish/phase/")
for i1, binoffset in enumerate(np.linspace(0, 2 * np.pi / number_of_bins, total_points // number_of_bins)):
    phases = np.mod([float(os.path.splitext(filename)[0]) + binoffset for filename in files], 2 * np.pi)
    indices = np.digitize(phases, bins)
    for i2, b in enumerate(bins):
        filestopiv = np.array(files)[indices == i2 + 1]
        if filestopiv.shape[0] > 0:
            piv = PIV(np.mod(b - binoffset, 2 * np.pi), 24, 24, 16, 0.55, "5pointgaussian", True)
            piv.add_video(["../data/zebrafish/phase/" + str(f) for f in filestopiv])
            piv.get_spaced_coordinates()
            piv.get_correlation_matrices()
            piv.get_correlation_averaged_velocity_field()
            piv.plot_flow_field(f"../analysis/visualisations/09022022/zebrafish_flowfield_moving_bin/{int((np.mod(b - binoffset, 2 * np.pi)) * 1e17)}.png")
