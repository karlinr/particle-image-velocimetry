import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import winsound

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rc('text', usetex=True)
plt.rcParams.update({"pgf.texsystem": "xelatex", "font.family": "serif", "text.usetex": True, 'pgf.rcfonts': False})


binsize = 31
files = os.listdir("../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(np.min(phases), np.max(phases), binsize)
indices = np.digitize(phases, bins)

for i, b in enumerate(bins):
    if i == 13:
        filestopiv = np.array(files)[indices == i + 1]
        piv = PIV(b, 24, 12, 1, 0.0, "9pointgaussian", False)
        piv.add_video(["../data/zebrafish/phase/" + str(f) for f in filestopiv])
        piv.set_coordinate(196, 234)
        #piv = PIV(f"", 44, 14, 1, 0, "9pointgaussian", False, True)
        #piv.add_video(f"../data/simulated/constant_report/0.tif")
        #piv.set_coordinate(36, 36)
        #piv = PIV(f"", 24, 12, 1, 0, "9pointgaussian", False, True)
        #piv.add_video(f"../data/simulated/outliers_report_4/0.tif")
        #piv.set_coordinate(36, 36)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        print(f"pass 1 - i: {i}, b: {b}, v: {piv.x_velocity_averaged()[0, 0]}")

        print(piv.x_velocity_averaged())

        plt.imshow(piv.correlation_averaged[0,0,0])
        plt.show()

        meanx = piv.x_velocity_averaged().flatten()[0]
        tmpx = []

        # Create our bootstrap distribution
        for j in range(50000):
            piv.resample()
            piv.get_correlation_averaged_velocity_field()
            tmpx.append(piv.x_velocity_averaged().flatten()[0])

        """plt.hist(tmpx, bins = 100)
        plt.show()"""

        median = np.median(tmpx)
        absdiffmedian = np.abs(tmpx - median)
        medianabsdev = np.median(absdiffmedian)

        zscore = 0.6745 * (tmpx - median) / medianabsdev

        print("Outliers")
        print(np.count_nonzero(zscore > 3.5))
        print(np.count_nonzero(zscore < -3.5))

        plt.figure(figsize = (3, 3))
        plt.hist(zscore, bins = 100, rasterized=True)
        plt.xlabel("Z-score")
        plt.ylabel("Frequency of occurence")
        plt.tight_layout()
        plt.savefig("outlier_z_plot.pgf")
        plt.show()

        tmpx = np.array(zscore)
        print(stats.kurtosis(tmpx))
        print(stats.skew(tmpx))
        kurts = []
        skews = []
        for _ in range(200):
            samplearg = np.random.choice(tmpx.shape[0], tmpx.shape[0])
            # print(samplearg)
            kurts.append(stats.kurtosis(tmpx[samplearg]))
            skews.append(stats.skew(tmpx[samplearg]))

        print(np.std(kurts))
        print(np.std(skews))

        print(zscore)
