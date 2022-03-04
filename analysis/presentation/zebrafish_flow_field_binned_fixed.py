import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rc('text', usetex=True)
plt.rcParams.update({"pgf.texsystem": "xelatex", "font.family": "serif", "text.usetex": True, 'pgf.rcfonts': False})


binsize = 50
files = os.listdir("../../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(np.min(phases), np.max(phases), binsize)
indices = np.digitize(phases, bins)

vs_x_averaged = []
vs_x_averaged_phases = []
vs_x_averaged_uncertainty = []
vs_x = []
vs_x_phases = []
for i, b in enumerate(bins):
    filestopiv = np.array(files)[indices == i + 1]
    piv = PIV(b, 64, 20, 1, 0.0, "5pointgaussian", False)
    piv.add_video(["../../data/zebrafish/phase/" + str(f) for f in filestopiv])
    piv.set_coordinate(196, 234)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    offset = piv.correlation_averaged_velocity_field
    piv.set(20, 16, 1)
    piv.set_coordinate(196, 234)
    piv.do_pass()
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs_x_averaged.append(piv.x_velocity_averaged()[0, 0])

    vs_x_averaged_uncertainty.append(piv.get_uncertainty(50)[0][0, 0])
    vs_x_averaged_phases.append(b + np.pi / binsize)
    for f in filestopiv:
        piv = PIV(b, 20, 16, 1, 0.0, "5pointgaussian", False)
        piv.add_video(f"../../data/zebrafish/phase/{f}")
        piv.set_coordinate(196, 234)
        piv.passoffset = offset.astype(int)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        vs_x.append(piv.x_velocity_averaged()[0, 0])
        vs_x_phases.append(float(os.path.splitext(f)[0]))

plt.figure(figsize = (3.2, 2.6))
plt.scatter(vs_x_phases, vs_x, s = 0.125, alpha = 0.5)
plt.errorbar(vs_x_averaged_phases, vs_x_averaged, yerr = vs_x_averaged_uncertainty, capsize = 1.5, capthick = 1, elinewidth = 1, ls = "None", color = "black")
plt.xlabel("Phase (rads)")
plt.ylabel("X Displacement (px)")
plt.tight_layout()
plt.savefig('analysis_distribution.pgf', transparent = True)
plt.show()
