from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# Systematically remove outliers from left and right to see if correlation averaged return to mean

# MPL
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

piv = PIV(f"", 24, 27, 24, 0, "5pointgaussian", False)
piv.add_video(f"../data/zebrafish/processed/23.tif")
piv.set_coordinate(201, 240)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
piv.get_velocity_field()

plt.hist(piv.x_velocity().flatten(), bins = 5)
plt.show()

bnum = []
vs_averaged = []
vs_averaged_std = []
vs_mean = []
vs_mean_std = []
bins = np.linspace(np.min(piv.x_velocity().flatten()), np.max(piv.y_velocity().flatten()), 5, endpoint = False)
print(bins)
indices = np.digitize(piv.x_velocity().flatten(), bins)
for i, b in enumerate(bins):
    bnum.append(i)
    if(np.any(indices == 1)):
        piv.resample_specific(indices == i + 1)
        piv.get_correlation_averaged_velocity_field()
        vs_averaged.append(piv.x_velocity_averaged().flatten()[0])
        tmp = []
        for i2 in range(100):
            piv.resample_from_array(indices == i2 + 1)
            piv.get_correlation_averaged_velocity_field()
            tmp.append(piv.x_velocity_averaged().flatten()[0])
        vs_averaged_std.append(np.std(tmp))
        vs_mean.append(np.mean(piv.x_velocity()[indices == i + 1]))
        vs_mean_std.append(np.std(piv.x_velocity()[indices == i + 1]) / len(piv.x_velocity()[indices == i + 1].flatten())**0.5)
    else:
        vs_averaged.append(np.nan)
        vs_mean.append(np.nan)

piv.resample_reset()
piv.get_correlation_averaged_velocity_field()
plt.axhline(piv.y_velocity_averaged().flatten()[0])
plt.errorbar(bnum, vs_averaged, yerr = vs_averaged_std, marker = "x", label = "correlation averaged", color = "red", ls = "none")
plt.errorbar(bnum, vs_mean, yerr = vs_mean_std, marker = "x", label = "bin averaged", ls = "none")
plt.legend()
plt.xlabel("Bin")
plt.ylabel("Displacement (px)")
plt.show()
