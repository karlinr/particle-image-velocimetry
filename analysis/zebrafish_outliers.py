from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# MPL
# plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


means = []
lowers = []
lowstd = []
uppers = []
upstd = []

for filename in os.listdir(f"../data/zebrafish/processed/"):
    piv = PIV(f"../data/zebrafish/processed/{filename}", 24, 27, 24, 0, "5pointgaussian", False)
    piv.set_coordinate(201, 240)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    piv.get_velocity_field()
    meanvelocity = piv.x_velocity_averaged()[0]
    means.append(meanvelocity)

    arr = np.flatnonzero(piv.x_velocity() > np.percentile(piv.x_velocity().flatten(), 68))
    piv.resample_specific(arr)
    piv.get_correlation_averaged_velocity_field()
    uppers.append(piv.x_velocity_averaged() - meanvelocity)
    up = []
    for i in range(100):
        piv.resample_from_array(arr)
        piv.get_correlation_averaged_velocity_field()
        up.append(piv.x_velocity_averaged() - meanvelocity)
    upstd.append(np.std(up, ddof = 1))

    arr = np.flatnonzero(piv.x_velocity() < np.percentile(piv.x_velocity().flatten(), 32))
    piv.resample_specific(arr)
    piv.get_correlation_averaged_velocity_field()
    lowers.append(piv.x_velocity_averaged() - meanvelocity)
    low = []
    for i in range(100):
        piv.resample_from_array(arr)
        piv.get_correlation_averaged_velocity_field()
        low.append(piv.x_velocity_averaged() - meanvelocity)
    lowstd.append(np.std(low, ddof = 1))


#plt.scatter(range(len(means)), means, s = 6, label = "Mean", marker = "_")
plt.errorbar(range(len(uppers)), uppers, yerr = upstd, ls = "None", label = "Upper")
plt.errorbar(range(len(lowers)), lowers, yerr = lowstd, ls = "None", label = "Lower")
plt.legend()
plt.show()
