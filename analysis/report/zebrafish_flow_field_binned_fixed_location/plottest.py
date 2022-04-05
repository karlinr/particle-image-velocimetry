import numpy as np
import matplotlib.pyplot as plt

phases = np.loadtxt("phases.txt")
phase_bins = np.loadtxt("phases-bins.txt")
correlation_averaged = np.loadtxt("correlation-averaged.txt")
non_correlation_averaged = np.loadtxt("non-correlation-averaged.txt")
correlation_averaged_se = np.loadtxt("correlation-averaged-se.txt")
non_correlation_averaged_mean = np.loadtxt("non-correlation-averaged-mean.txt")
non_correlation_averaged_mean_se = np.loadtxt("non-correlation-averaged-mean-se.txt")

plt.scatter(phases, non_correlation_averaged, s = 1, c ="black")
plt.errorbar(phase_bins, correlation_averaged, yerr = correlation_averaged_se)
plt.errorbar(phase_bins, non_correlation_averaged_mean, yerr = non_correlation_averaged_mean_se)
plt.show()