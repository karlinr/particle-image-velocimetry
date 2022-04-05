import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import skew, kurtosis

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rc('text', usetex=True)
plt.rcParams.update({"pgf.texsystem": "xelatex", "font.family": "serif", "text.usetex": True, 'pgf.rcfonts': False})


binsize = 31
files = os.listdir("../../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(np.min(phases), np.max(phases), binsize)
indices = np.digitize(phases, bins)

for i, b in enumerate(bins):
    if i == 14:
        filestopiv = np.array(files)[indices == i + 1]
        piv = PIV(b, 24, 24, 1, 0.0, "9pointgaussian", False)
        piv.add_video(["../../data/zebrafish/phase/" + str(f) for f in filestopiv])
        piv.set_coordinate(196, 234)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        print(f"pass 1 - i: {i}, b: {b}, v: {piv.x_velocity_averaged()[0, 0]}")

        tmpx = []
        tmpy = []
        for _ in range(2500):
            piv.resample(100)
            piv.get_correlation_averaged_velocity_field()
            tmpx.append(piv.x_velocity_averaged().flatten()[0])
            tmpy.append(piv.y_velocity_averaged().flatten()[0])
        print(np.std(tmpx))

        piv.resample_reset()
        piv.get_correlation_averaged_velocity_field()

        plt.figure(figsize = (3.2, 2.4))
        plt.hist2d(tmpx, tmpy, bins = 60, rasterized = True)
        plt.xlabel("X Displacement (px)")
        plt.ylabel("Y Displacement (px)")
        plt.tight_layout()
        plt.savefig('analysis_distribution_bootstrapped_2d.pgf', transparent = True)
        plt.show()

        print(f"X - Skew: {skew(tmpx)}, Kurtosis: {skew(tmpx)}")
        plt.figure(figsize = (2.3, 2))
        plt.hist(tmpx, bins = 200)
        plt.axvline(piv.x_velocity_averaged().flatten()[0], color = "tab:orange")
        plt.xlabel("X Displacement (px)")
        plt.ylabel("Frequency of Occurence")
        plt.tight_layout()
        plt.savefig('analysis_distribution_bootstrapped_x.pgf', transparent = True)
        plt.show()

        print(f"Y - Skew: {skew(tmpy)}, Kurtosis: {skew(tmpy)}")
        plt.figure(figsize = (2.3, 2))
        plt.hist(tmpy, bins = 200)
        plt.axvline(piv.y_velocity_averaged().flatten()[0], color = "tab:orange")
        plt.xlabel("Y Displacement (px)")
        plt.ylabel("Frequency of Occurence")
        plt.tight_layout()
        plt.savefig('analysis_distribution_bootstrapped_y.pgf', transparent = True)
        plt.show()