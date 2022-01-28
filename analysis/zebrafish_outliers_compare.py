from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# MPL
# plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

piv = PIV(f"", 24, 27, 24, 0, "5pointgaussian", False)
piv.add_video(f"../data/zebrafish/processed/22.tif")
piv.set_coordinate(201, 240)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
piv.get_velocity_field()
v_temp = []
for i in range(100):
    piv.resample()
    piv.get_correlation_averaged_velocity_field()
    v_temp.append(piv.x_velocity_averaged().flatten()[0])
std = np.std(v_temp, ddof = 1)


percs = []
uppers = []
uppers_std = []
for percentage in range(70):
    arr = np.flatnonzero(piv.x_velocity() >= np.percentile(piv.x_velocity().flatten(), percentage))
    piv.resample_specific(arr)
    piv.get_correlation_averaged_velocity_field()
    percs.append(percentage)
    uppers.append(piv.x_velocity_averaged().flatten()[0])
    v_temp = []
    for i in range(200):
        piv.resample_from_array(arr)
        piv.get_correlation_averaged_velocity_field()
        v_temp.append(piv.x_velocity_averaged().flatten()[0])
    uppers_std.append(np.std(v_temp, ddof = 1))
lowers = []
lowers_std = []
for percentage in range(70):
    arr = np.flatnonzero(piv.x_velocity() <= np.percentile(piv.x_velocity().flatten(), 100 - percentage))
    piv.resample_specific(arr)
    piv.get_correlation_averaged_velocity_field()
    lowers.append(piv.x_velocity_averaged().flatten()[0])
    v_temp = []
    for i in range(200):
        piv.resample_specific(arr[np.random.choice(len(arr), len(arr))])
        piv.get_correlation_averaged_velocity_field()
        v_temp.append(piv.x_velocity_averaged().flatten()[0])
    lowers_std.append(np.std(v_temp, ddof = 1))

mids = []
mids_std = []
for percentage in range(70):
    arr = np.flatnonzero(np.logical_and(piv.x_velocity().flatten() <= np.percentile(piv.x_velocity().flatten(), 100 - percentage / 2), piv.x_velocity().flatten() >= np.percentile(piv.x_velocity().flatten(), percentage / 2)))
    piv.resample_specific(arr)
    piv.get_correlation_averaged_velocity_field()
    mids.append(piv.x_velocity_averaged().flatten()[0])
    v_temp = []
    for i in range(200):
        piv.resample_specific(arr[np.random.choice(len(arr), len(arr))])
        piv.get_correlation_averaged_velocity_field()
        v_temp.append(piv.x_velocity_averaged().flatten()[0])
    mids_std.append(np.std(v_temp, ddof = 1))

piv.resample_reset()
piv.get_correlation_averaged_velocity_field()
plt.axhline(piv.x_velocity_averaged().flatten()[0], c = "black")
plt.axhline(piv.x_velocity_averaged().flatten()[0] + std, c = "black", ls = ":", lw = 1)
plt.axhline(piv.x_velocity_averaged().flatten()[0] - std, c = "black", ls = ":", lw = 1)
plt.plot(percs, mids, c = "black", label = "Mean")
plt.fill_between(percs, np.array(mids) - np.array(mids_std), np.array(mids) + np.array(mids_std), color = "black", alpha = 0.2)
plt.plot(percs, uppers, c = "blue", label = "Upper")
plt.fill_between(percs, np.array(uppers) - np.array(uppers_std), np.array(uppers) + np.array(uppers_std), color = "blue", alpha = 0.2)
plt.plot(percs, lowers, c = "orange", label = "Lower")
plt.fill_between(percs, np.array(lowers) - np.array(lowers_std), np.array(lowers) + np.array(lowers_std), color = "orange", alpha = 0.2)
plt.xlabel("Excluded (%)")
plt.ylabel("X Displacement (px)")
plt.legend()
plt.show()
