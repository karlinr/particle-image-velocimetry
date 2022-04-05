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
kurtosis_with = []
skew_with = []
num_outliers_with =[]
kurtosis_without = []
skew_without = []
num_outliers_without =[]

for i, filename in enumerate(os.listdir(f"../../../data/simulated/outliers/")):
    print(i)
    # Do PIV
    piv = PIV(f"", 24, 12, 1, 0, "9pointgaussian", False, True)
    piv.add_video(f"../../../data/simulated/outliers/{filename}")
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

    kurtosis_with.append(stats.kurtosis(tmp))
    skew_with.append(stats.skew(tmp))
    # Get modified z-score
    median = np.median(tmp)
    absdiffmedian = np.abs(tmp - median)
    medianabsdev = np.median(absdiffmedian)
    zscore = 0.6745 * (tmp - median) / medianabsdev
    num_outliers_with.append(np.count_nonzero(zscore > 3.5))

for i, filename in enumerate(os.listdir(f"../../../data/simulated/no_outliers/")):
    print(i)
    # Do PIV
    piv = PIV(f"", 24, 12, 1, 0, "9pointgaussian", False, True)
    piv.add_video(f"../../../data/simulated/no_outliers/{filename}")
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

    kurtosis_without.append(stats.kurtosis(tmp))
    skew_without.append(stats.skew(tmp))
    # Get modified z-score
    median = np.median(tmp)
    absdiffmedian = np.abs(tmp - median)
    medianabsdev = np.median(absdiffmedian)
    zscore = 0.6745 * (tmp - median) / medianabsdev
    num_outliers_without.append(np.count_nonzero(zscore > 3.5))

print("kurtosis")
print(np.mean(kurtosis_with))
print(np.std(kurtosis_with) / len(kurtosis_with)**0.5)
print(np.mean(kurtosis_without))
print(np.std(kurtosis_without) / len(kurtosis_without)**0.5)
stat, p = stats.ttest_ind(kurtosis_with, kurtosis_without, alternative = "greater")
print(p)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

print("skew")
print(np.mean(skew_with))
print(np.std(skew_with) / len(skew_with)**0.5)
print(np.mean(skew_without))
print(np.std(skew_without) / len(skew_without)**0.5)
stat, p = stats.ttest_ind(skew_with, skew_without, alternative = "greater")
plt.hist(skew_with, bins = 100)
plt.show()
plt.hist(skew_without, bins = 100)
plt.show()
print(p)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

print("outliers")
print(np.mean(num_outliers_with))
print(np.std(num_outliers_with) / len(num_outliers_with)**0.5)
print(np.mean(num_outliers_without))
print(np.std(num_outliers_without) / len(num_outliers_without)**0.5)
stat, p = stats.ttest_ind(num_outliers_with, num_outliers_without, alternative = "greater")
print('stat=%.3f, p=%.3f' % (stat, p))
print(p)
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')