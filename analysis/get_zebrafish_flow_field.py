from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# MPL
#plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Setup Arrays
phases = []
vels = []
stderrs = []

for filename in os.listdir(f"../data/zebrafish/processed/"):
    # Setup PIV
    piv = PIV(f"../data/zebrafish/processed/{filename}", 24, 24, 16, 0.6, "5pointgaussian", False)
    piv.set_coordinate(190, 238)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()

    # Get PIV output
    phases.append(int(os.path.splitext(os.path.basename(filename))[0]) / 31 * np.pi * 2.0)
    vels.append(piv.x_velocity_averaged()[0, 0])

    # Get standard error
    vels_temp = []
    for i in range(500):
        piv.resample()
        piv.get_correlation_averaged_velocity_field()
        vels_temp.append(piv.y_velocity_averaged()[0, 0])
    stderrs.append(np.std(vels_temp, ddof = 1))

# Plot it
xs, ys, err = zip(*sorted(zip(phases, vels, stderrs)))
plt.figure(figsize = (8, 8))
plt.axhline(0, c = "black", ls = ":", lw = 0.5)
plt.plot(xs, ys, c = "black", ls = "--", lw = 0.5)
plt.errorbar(xs, ys, yerr = err, capsize = 3, capthick = 1, elinewidth = 1, ls = "None", c = "black")
plt.scatter(xs, ys, c = "black", s = 6)
plt.xlabel("Phase (rads)")
plt.ylabel("X displacement (px)")
plt.xticks(xs, rotation = 60)
plt.tight_layout()
"""for x in np.linspace(0, 2 * np.pi, 31, endpoint = False):
    plt.axvline(x - 1/31 * np.pi, lw = 0.2, ls = ":", c = "black")"""
plt.show()
