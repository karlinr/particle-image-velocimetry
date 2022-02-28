from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# Plots unbinned data + binned data for give binsizes
# Uses dot product to find vector similarity

# MPL
#plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

files = os.listdir("../data/zebrafish/phase/")

vs = []
phases = []
plt.figure(figsize = (8, 8))

binsize = 31
files = os.listdir("../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
print(phases)
bins = np.linspace(np.min(phases), np.max(phases), binsize)
indices = np.digitize(phases, bins)

vs = []
vs_unb = []
phases = []
vs_ub_phases = []

for i, b in enumerate(bins):
    filestopiv = np.array(files)[indices == i + 1]
    piv = PIV(b, 24, 24, 24, 0.6, "5pointgaussian", False)
    piv.add_video(["../data/zebrafish/phase/" + str(f) for f in filestopiv])
    if i == 23:
        plt.imshow(piv.intensity_array_for_display)
        plt.show()
    piv.set_coordinate(198, 234)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs.append(piv.x_velocity_averaged()[0, 0])
    v = [piv.x_velocity_averaged()[0, 0], piv.y_velocity_averaged()[0, 0]]
    #piv.plot_flow_field()
    #print(v)
    phases.append(b + (np.pi) / binsize)
    for f in filestopiv:
        piv = PIV(b, 24, 24, 24, 0.6, "5pointgaussian", False)
        piv.add_video(f"../data/zebrafish/phase/{f}")
        piv.set_coordinate(201, 240)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        vs_unb.append(piv.x_velocity_averaged()[0, 0])
        vs_ub_phases.append(float(os.path.splitext(f)[0]))

plt.scatter(vs_ub_phases, vs_unb, s = 1, c = "black")
plt.scatter(phases, vs, label = f"Bins : {binsize}", c = "red")
plt.xlabel("Phase (Rads)")
plt.ylabel("Displacement (px)")
plt.legend()
plt.show()