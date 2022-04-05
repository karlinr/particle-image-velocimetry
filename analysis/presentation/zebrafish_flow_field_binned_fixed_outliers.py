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
        piv.get_velocity_field()
        true_x_velocity = piv.x_velocity_averaged().flatten()[0]
        true_x_velocity_std = piv.get_uncertainty(1000)[0]

        indices = np.argwhere(piv.x_velocity()[:, 0, 0] < -5).flatten()
        piv.resample_specific(indices)
        piv.get_correlation_averaged_velocity_field()
        outlier_x_velocity = piv.x_velocity_averaged()
        tmp = []
        for _ in range(1000):
            piv.resample_from_array(indices)
            piv.get_correlation_averaged_velocity_field()
            tmp.append(piv.x_velocity_averaged().flatten()[0])
        outlier_x_velocity_std = np.std(tmp)


        piv.resample_reset()
        piv.get_correlation_averaged_velocity_field()
        piv.get_velocity_field()
        plt.figure(figsize = (2.6, 2.3))
        plt.hist(piv.x_velocity().flatten(), bins = 20)
        plt.axvline(true_x_velocity, color = "tab:orange")
        #plt.axvline(true_x_velocity + true_x_velocity_std, color = "tab:orange", ls = ":")
        #plt.axvline(true_x_velocity - true_x_velocity_std, color = "tab:orange", ls = ":")
        plt.axvline(outlier_x_velocity, color = "tab:green")
        #plt.axvline(outlier_x_velocity + outlier_x_velocity_std, color = "tab:green", ls = ":")
        #plt.axvline(outlier_x_velocity - outlier_x_velocity_std, color = "tab:green", ls = ":")
        plt.axvline(np.mean(piv.x_velocity()[indices]), color = "black")
        print(np.std(piv.x_velocity()[indices], ddof = 1) / np.sqrt(len(indices)))
        plt.xlabel("X Displacement (px)")
        plt.ylabel("Frequency of Occurence")
        plt.tight_layout()
        plt.savefig("analysis_outliers.pgf", transparent = True)
        plt.show()