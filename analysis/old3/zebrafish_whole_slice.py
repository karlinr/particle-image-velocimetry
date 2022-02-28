from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# Compare each set of phases averaged velocities over one cycle compared to full dataset

# MPL
#plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

files = os.listdir("../../data/zebrafish/phase/")
piv = PIV("Full", 24, 24, 24, 0, "5pointgaussian", False)
piv.add_video([f"../data/zebrafish/phase/" + str(f) for f in files])
piv.set_coordinate(201, 240)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
v_mean = piv.x_velocity_averaged().flatten()[0]
v_temp = []
for i in range(500):
    piv.resample()
    piv.get_correlation_averaged_velocity_field()
    v_temp.append(piv.x_velocity_averaged().flatten()[0])
v_mean_std = np.std(v_temp)

folders = os.listdir("../../data/zebrafish/phasetime/")

vs = []
vs_std = []

for folder in folders:
    files = os.listdir(f"../data/zebrafish/phasetime/{folder}/")
    piv = PIV(f"{folder}", 24, 24, 24, 0, "5pointgaussian", False)
    piv.add_video([f"../data/zebrafish/phasetime/{folder}/" + str(f) for f in files])
    piv.set_coordinate(201, 240)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs.append(piv.x_velocity_averaged().flatten()[0])
    v_temp = []
    for i in range(500):
        piv.resample()
        piv.get_correlation_averaged_velocity_field()
        v_temp.append(piv.x_velocity_averaged().flatten()[0])
    vs_std.append(np.std(v_temp))

plt.hist(vs, bins = 30)
plt.axvline(v_mean, c = "black", lw = 1)
plt.axvline(v_mean - v_mean_std, c = "black", ls = ":", lw = 1)
plt.axvline(v_mean + v_mean_std, c = "black", ls = ":", lw = 1)
plt.show()

plt.errorbar(range(len(vs)), vs, yerr = vs_std, ls="none")
plt.axhline(v_mean)
plt.show()

errstomean = (v_mean - vs) / vs_std
plt.hist(errstomean, bins = 30)
plt.show()

print(np.std(errstomean))
print(np.mean(errstomean))