import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rc('text', usetex=True)
plt.rcParams.update({"pgf.texsystem": "xelatex", "font.family": "serif", "text.usetex": True, 'pgf.rcfonts': False})


binsize = 31
files = os.listdir("../../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(-1e-32, 2 * np.pi-1e-32, binsize)
print(bins)
indices = np.digitize(phases, bins)
print(indices)
print(np.unique(indices))

vs_x_averaged = []
vs_x_averaged_phases = []
vs_x_averaged_uncertainty = []
vs_x = []
vs_x_phases = []
for b in np.unique(indices):
    if b == 14:
        print(b)
        filestopiv = np.array(files)[indices == b]
        piv = PIV(b, 64, 24, 1, 0.0, "5pointgaussian", False)
        piv.add_video(["../../data/zebrafish/phase/" + str(f) for f in filestopiv])
        piv.set_coordinate(196, 234)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.begin_draw()
        piv.draw_intensity()
        piv.draw_iw()
        piv.draw_sa()
        scalebar = ScaleBar(2.202643171806167, "um", length_fraction = 0.25, box_color = None, box_alpha = 0, color="white")
        plt.gca().add_artist(scalebar)
        plt.tight_layout()
        plt.savefig("analysis_region1.pgf")
        piv.end_draw()
        print(f"pass 1 - b: {b}, v: {piv.x_velocity_averaged()[0, 0]}")
        piv.set(24, 8, 1)
        piv.do_pass()
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        print(f"pass 2 - b: {b}, v: {piv.x_velocity_averaged()[0, 0]}")
        piv.begin_draw()
        piv.draw_intensity()
        piv.draw_iw()
        piv.draw_sa()
        scalebar = ScaleBar(2.202643171806167, "um", length_fraction = 0.25, box_color = None, box_alpha = 0, color="white")
        plt.gca().add_artist(scalebar)
        plt.tight_layout()
        plt.savefig("analysis_region2.pgf")
        piv.end_draw()
        offset = piv.correlation_averaged_velocity_field
        vs_x_averaged.append(piv.x_velocity_averaged()[0, 0])

        vs_x_averaged_uncertainty.append(piv.get_uncertainty(1000)[0][0, 0])
        vs_x_averaged_phases.append(bins[b] - (np.pi) / (binsize - 1))
        print(vs_x_averaged_phases)
        for f in filestopiv:
            piv = PIV(b, 64, 24, 1, 0.0, "5pointgaussian", False)
            piv.add_video(f"../../data/zebrafish/phase/{f}")
            piv.set_coordinate(196, 234)
            #piv.passoffset = np.round(offset)
            piv.get_correlation_matrices()
            piv.get_correlation_averaged_velocity_field()
            piv.set(64, 24, 1)
            piv.do_pass()
            piv.get_correlation_matrices()
            piv.get_correlation_averaged_velocity_field()
            piv.set(24, 8, 1)
            piv.do_pass()
            piv.get_correlation_matrices()
            piv.get_correlation_averaged_velocity_field()
            vs_x.append(piv.x_velocity_averaged()[0, 0])
            vs_x_phases.append(float(os.path.splitext(f)[0]))

plt.figure(figsize = (3.2, 2.6))
plt.scatter(vs_x_phases, vs_x, marker = "o", s = 0.2, alpha = 1, facecolors = "white", edgecolors = "royalblue", linewidths=0.1)
#plt.scatter(vs_x_phases, vs_x, marker = "o", s = 0.125, alpha = 1, linewidths=0.15, c = vs_x_phases, cmap = "viridis")
plt.errorbar(vs_x_averaged_phases, vs_x_averaged, yerr = vs_x_averaged_uncertainty, capsize = 1.5, capthick = 1, elinewidth = 1, ls = "None", color = "black")
plt.xlabel("Phase (rads)")
plt.ylabel("X Displacement (px)")
plt.tight_layout()
#plt.savefig('analysis_distribution.pgf', transparent = True)
plt.show()
