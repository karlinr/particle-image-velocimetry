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
files = os.listdir("../../../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(np.min(phases), np.max(phases), binsize)
indices = np.digitize(phases, bins)

for b in np.unique(indices):
    if b == 14:
        filestopiv = np.array(files)[indices == b]
        piv = PIV(b, 24, 24, 1, 0.0, "9pointgaussian", False, True)
        piv.add_video(["../../../data/zebrafish/phase/" + str(f) for f in filestopiv])
        piv.set_coordinate(196, 234)
        #piv = PIV(f"", 24, 24, 1, 0, "5pointgaussian", False)
        #piv.add_video(f"../../data/simulated/constant_for_presentation/0.tif")
        #piv.set_coordinate(36, 36)
        #piv = PIV(f"", 24, 12, 1, 0, "9pointgaussian", False, True)
        #piv.add_video(f"../../../data/simulated/outliers_report1/0.tif")
        #piv.set_coordinate(36, 36)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        print(f"pass 1, b: {b}, v: {piv.x_velocity_averaged()[0, 0]}")

        plt.imshow(piv.correlation_averaged[0, 0, 0])
        plt.show()

        meanx = piv.x_velocity_averaged().flatten()[0]
        tmpx = []

        for _ in range(50000):
            piv.resample()
            piv.get_correlation_averaged_velocity_field()
            tmpx.append(piv.x_velocity_averaged().flatten()[0])

        tmpx = np.array(tmpx)
        print(stats.kurtosis(tmpx))
        print(stats.skew(tmpx))
        kurts = []
        skews = []
        for _ in range(1000):
            samplearg = np.random.choice(tmpx.shape[0], tmpx.shape[0])
            # print(samplearg)
            kurts.append(stats.kurtosis(tmpx[samplearg]))
            skews.append(stats.skew(tmpx[samplearg]))

        print(np.std(kurts))
        print(np.std(skews))

        plt.figure(figsize = (3, 3))
        plt.hist(tmpx, bins = 100, rasterized=True)
        plt.xlabel("X-displacement (px)")
        plt.ylabel("Frequency of occurence")
        plt.tight_layout()
        plt.savefig("sim_dist_outlier.pgf")
        plt.show()

        tmpxbootlierright = []
        tmpxbootlierleft = []
        tmpxbootlierboth = []
        for j in range(5000):
            tmpx = []
            for k in range(100):
                piv.resample()
                piv.get_correlation_averaged_velocity_field()
                tmpx.append(piv.x_velocity_averaged().flatten()[0])
            tmpx = sorted(tmpx)
            tmpxbootlierright.append(meanx - np.mean(tmpx[:-4]))
            tmpxbootlierleft.append(meanx - np.mean(tmpx[4:]))
            tmpxbootlierboth.append(meanx - np.mean(tmpx[2:-2]))

        """plt.hist(tmpxbootlierright, bins = 30)
        plt.xlabel("Trimmed mean - mean")
        plt.ylabel("Frequency of occurence")
        plt.show()

        plt.hist(tmpxbootlierleft, bins = 30)
        plt.xlabel("Trimmed mean - mean")
        plt.ylabel("Frequency of occurence")
        plt.show()"""

        plt.figure(figsize = (3, 3))
        plt.hist(tmpxbootlierboth, bins = 50, rasterized=True)
        plt.xlabel("Trimmed mean - mean")
        plt.ylabel("Frequency of occurence")
        plt.tight_layout()
        plt.savefig("bootlierboth_sim_with.pgf")
        plt.show()

        duration = 100000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)
