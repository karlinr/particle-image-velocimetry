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

vels = []
outs = []
numoutliers_with = []
numoutliers_without = []

for i, filename in enumerate(os.listdir(f"../../data/simulated/outliers/")):
    print(i)
    # Do PIV
    piv = PIV(f"", 24, 12, 1, 0, "9pointgaussian", False, True)
    piv.add_video(f"../../data/simulated/outliers/{filename}")
    piv.set_coordinate(36, 36)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()

    # Get distribution
    tmp = []
    # Create our bootstrap distribution
    for j in range(200):
        piv.resample()
        piv.get_correlation_averaged_velocity_field()
        tmp.append(piv.x_velocity_averaged().flatten()[0])

    # Get modified z-score
    median = np.median(tmp)
    absdiffmedian = np.abs(tmp - median)
    medianabsdev = np.median(absdiffmedian)
    zscore = 0.6745 * (tmp - median) / medianabsdev

    # Count number of outliers
    #numoutliers_with.append(np.count_nonzero(zscore > 3.5))
    numoutliers_with.append(stats.skew(zscore))

print(np.mean(numoutliers_with))
print(np.std(numoutliers_with) / len(numoutliers_with)**0.5)

for i, filename in enumerate(os.listdir(f"../../data/simulated/no_outliers/")):
    print(i)
    # Do PIV
    piv = PIV(f"", 24, 8, 1, 0, "9pointgaussian", False, True)
    piv.add_video(f"../../data/simulated/outliers/{filename}")
    piv.set_coordinate(36, 36)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()

    # Get distribution
    tmp = []
    # Create our bootstrap distribution
    for j in range(200):
        piv.resample()
        piv.get_correlation_averaged_velocity_field()
        tmp.append(piv.x_velocity_averaged().flatten()[0])

    # Get modified z-score
    median = np.median(tmp)
    absdiffmedian = np.abs(tmp - median)
    medianabsdev = np.median(absdiffmedian)
    zscore = 0.6745 * (tmp - median) / medianabsdev

    # Count number of outliers
    #numoutliers_without.append(np.count_nonzero(zscore > 3.5))
    numoutliers_without.append(stats.skew(zscore))

print(np.mean(numoutliers_without))
print(np.std(numoutliers_without) / len(numoutliers_without)**0.5)

stat, p = stats.ttest_ind(numoutliers_with, numoutliers_without)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')