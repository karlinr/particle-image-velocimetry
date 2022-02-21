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
            piv = PIV(np.mod(b - binoffset, 2 * np.pi), 6, 20, 6, 0.35, "5pointgaussian", False)
            piv.add_video(["../data/zebrafish/phase/" + str(f) for f in filestopiv])
            piv.get_spaced_coordinates()
            piv.get_correlation_matrices()
            tmp = []
            for i in range(25):
                print(i)
                piv.resample()
                piv.get_correlation_averaged_velocity_field()
                tmp.append(piv.velocity_magnitude_averaged())
            std = np.std(tmp, axis = 0)
            piv.resample_reset()
            piv.get_correlation_averaged_velocity_field()
            U = piv.x_velocity_averaged()[:, :]
            V = piv.y_velocity_averaged()[:, :]
            mag = np.sqrt(U ** 2 + V ** 2)
            plt.figure(figsize = (12, 7))
            plt.title(piv.title)
            #plt.imshow(np.flip(np.flip(np.rot90(piv.intensity_array_for_display), axis = 1)), cmap = "gray", aspect = "auto")
            plt.pcolor(piv.coordinates[:, :, 0] + piv.xoffset + piv.sa + 0.5 * piv.iw - piv.inc / 2, piv.coordinates[:, :, 1] + piv.yoffset + piv.sa + 0.5 * piv.iw - piv.inc / 2, std / mag, cmap = "gray")
            plt.clim(0, 1)
            plt.colorbar()
            plt.quiver(piv.xcoords(), piv.ycoords(), U / mag, V / mag, mag, angles = "xy", scale_units = "xy", scale = 0.5)
            plt.savefig(f"../analysis/visualisations/09022022/zebrafish_flowfield_small_iw/{int((np.mod(b - binoffset, 2 * np.pi)) * 1e17)}.png")
            plt.show()
            #piv.plot_flow_field(f"../analysis/visualisations/09022022/zebrafish_flowfield_moving_bin/{int((np.mod(b - binoffset, 2 * np.pi)) * 1e17)}.png")
            print("Done")
