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
numoutliers = []

for i, folder in enumerate(os.listdir(f"../../data/simulated/outliers_random/")):
    print(i)
    # Get velocity and outliers
    vels.append(float(folder.split("_")[0]))
    outs.append(float(folder.split("_")[1]))

    # Do PIV
    piv = PIV(f"", 24, 14, 1, 0, "9pointgaussian", False, True)
    piv.add_video(f"../../data/simulated/outliers_random/{folder}/0.tif")
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

    # Get modified z-score
    median = np.median(tmp)
    absdiffmedian = np.abs(tmp - median)
    medianabsdev = np.median(absdiffmedian)
    zscore = 0.6745 * (tmp - median) / medianabsdev

    # Count number of outliers
    numoutliers.append(np.count_nonzero(zscore > 3.5))

plt.scatter(outs, numoutliers, s = 2)
plt.show()

plt.scatter(vels, numoutliers, s = 2)
plt.show()

print(stats.pearsonr(outs, numoutliers))
print(stats.pearsonr(vels, numoutliers))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(vels, outs, numoutliers)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()