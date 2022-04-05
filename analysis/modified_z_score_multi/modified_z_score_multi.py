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


outlier_size = []
numoutliers_mean = []
numoutliers_mean_error = []

for i in range(10):
    ktmp = []
    stmp = []
    numoutliers = []
    for filename in os.listdir(f"../../data/simulated/outliers_report_{i}/"):
        piv = PIV(f"", 24, 14, 1, 0, "9pointgaussian", False, True)
        piv.add_video(f"../../data/simulated/outliers_report_{i}/{filename}")
        piv.set_coordinate(36, 36)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()

        # Get distribution
        tmp = []
        # Create our bootstrap distribution
        for j in range(1000):
            piv.resample()
            piv.get_correlation_averaged_velocity_field()
            tmp.append(piv.x_velocity_averaged().flatten()[0])

        # Get kurtosis and skew
        #ktmp.append(stats.kurtosis(tmp))
        #stmp.append(stats.skew(tmp))

        # Get modified z-score
        median = np.median(tmp)
        absdiffmedian = np.abs(tmp - median)
        medianabsdev = np.median(absdiffmedian)
        zscore = 0.6745 * (tmp - median) / medianabsdev

        numoutliers.append(np.count_nonzero(zscore > 3.5))
    outlier_size.append(i)
    numoutliers_mean.append(np.mean(numoutliers))
    numoutliers_mean_error.append((np.std(numoutliers) / np.sqrt(100)))


plt.errorbar(outlier_size, numoutliers_mean, yerr = numoutliers_mean_error)
plt.show()
