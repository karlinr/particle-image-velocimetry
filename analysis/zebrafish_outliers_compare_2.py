from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# Take percentiles of data and see if correlation averaged result differs from percentile
# Subtracts from percentile
# > 0 within percentile
# < 0 outside percentile

# MPL
# plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

piv = PIV(f"", 24, 27, 24, 0, "5pointgaussian", False)
piv.add_video(f"../data/zebrafish/processed/15.tif")
piv.set_coordinate(201, 240)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
piv.get_velocity_field()

percs_low = []
percs_high = []
percs_mean = []
uppers = []
uppers_std = []
lowers = []
lowers_std = []
percentiles_low = []
percentiles_high = []
for percentage in range(0, 99):
    print(piv.x_velocity().flatten())
    print(np.percentile(piv.x_velocity().flatten(), percentage))
    print(piv.x_velocity().flatten() >= np.percentile(piv.x_velocity().flatten(), percentage))
    arr = np.flatnonzero(piv.x_velocity() >= np.percentile(piv.x_velocity().flatten(), percentage))
    piv.resample_specific(arr)
    piv.get_correlation_averaged_velocity_field()
    percs_high.append(percentage)

    percentiles_high.append(np.percentile(piv.x_velocity().flatten(), percentage))
    uppers.append(piv.x_velocity_averaged().flatten()[0])

    v_temp = []
    for i in range(200):
        piv.resample_from_array(arr)
        piv.get_correlation_averaged_velocity_field()
        v_temp.append(piv.x_velocity_averaged().flatten()[0])
    uppers_std.append(np.std(v_temp, ddof = 1))

for percentage in range(0, 99):
    arr = np.flatnonzero(piv.x_velocity() <= np.percentile(piv.x_velocity().flatten(), 100 - percentage))
    piv.resample_specific(arr)
    piv.get_correlation_averaged_velocity_field()

    percs_low.append(percentage)

    percentiles_low.append(np.percentile(piv.x_velocity().flatten(), 100 - percentage))
    lowers.append(piv.x_velocity_averaged().flatten()[0])

    v_temp = []
    for i in range(200):
        piv.resample_from_array(arr)
        piv.get_correlation_averaged_velocity_field()
        v_temp.append(piv.x_velocity_averaged().flatten()[0])
    lowers_std.append(np.std(v_temp, ddof = 1))

ys = np.array(uppers) - np.array(percentiles_high)
plt.plot(percs_high, ys, color = "blue")
plt.fill_between(percs_high, ys - uppers_std, ys + uppers_std, color = "blue", alpha = 0.1)

ys = np.array(percentiles_low) - np.array(lowers)
plt.plot(percs_low, ys, color = "orange")
plt.fill_between(percs_low, ys - lowers_std, ys + lowers_std, color = "orange", alpha = 0.1)

plt.axhline(0, color = "black", lw = 1, ls = ":")
plt.show()
