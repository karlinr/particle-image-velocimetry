import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


from scipy.stats import t
fig, ax = plt.subplots(1, 1)


doplot = True
plt.rc('text', usetex=True)
plt.rcParams.update({"pgf.texsystem": "xelatex", "font.family": "serif", "text.usetex": True, 'pgf.rcfonts': False})

zratio = []
kurtosis = []
skewness = []
means = []
stds = []

print(stats.t.rvs(df = 2499, loc = 0, scale = 80))

#its = [100, 250, 500, 1000]
its = [1000]
for iterations in its:
    vel_x_measured = []
    vel_x_std = []
    std_errs_to_mean = []
    for video in os.listdir("../../../data/simulated/constant_report/"):
        print(video)
        piv = PIV(f"", 44, 8, 1, 0, "9pointgaussian", False, True)
        piv.add_video(f"../../../data/simulated/constant_report/{video}")
        piv.set_coordinate(36, 36)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        vel_x_measured.append(piv.x_velocity_averaged().flatten()[0])
        vel_x_tmp = []
        for _ in range(iterations):
            piv.resample()
            piv.get_correlation_averaged_velocity_field()
            vel_x_tmp.append(piv.x_velocity_averaged().flatten()[0])
        if doplot and iterations == 1000:
            if video == "1.tif":
                plt.figure(figsize = (3, 3))
                plt.hist(vel_x_tmp, bins = 40, rasterized=True)
                plt.xlabel("$x$-displacement (px)")
                plt.ylabel("Frequency of occurence")
                plt.axvline(np.mean(vel_x_tmp) - np.std(vel_x_tmp), color = "r", lw = 3, alpha = 0.6)
                plt.axvline(np.mean(vel_x_tmp) + np.std(vel_x_tmp), color = "r", lw = 3, alpha = 0.6)
                plt.tight_layout()
                plt.savefig("bootstrapped_distribution.pgf")
                plt.show()
                doplot = False
        vel_x_std.append(np.std(vel_x_tmp))
    std_errs_to_mean = (vel_x_measured - np.mean(vel_x_measured)) / vel_x_std

    print(stats.normaltest(std_errs_to_mean))
    print(stats.kurtosis(std_errs_to_mean))
    print(stats.skew(std_errs_to_mean))
    print(np.std(std_errs_to_mean))

    kurts = []
    skews = []
    stds = []
    for _ in range(1000):
        samplearg = np.random.choice(std_errs_to_mean.shape[0], std_errs_to_mean.shape[0])
        #print(samplearg)
        kurts.append(stats.kurtosis(std_errs_to_mean[samplearg]))
        skews.append(stats.skew(std_errs_to_mean[samplearg]))
        stds.append(np.std(std_errs_to_mean[samplearg]))

    print(np.std(kurts))
    print(np.std(skews))
    print(np.std(stds))

    print("=====================")
    print(iterations)
    print(np.mean(vel_x_std))
    print(np.std(vel_x_measured))
    print("STD")
    print(np.std(vel_x_std) / len(vel_x_std)**0.5)
    vel_x_measured = np.array(vel_x_measured)
    vel_x_measured_tmp = []
    vel_x_measured_tmp_std = []
    for i in range(2500):
        samplearg = np.random.choice(2500, 2500)
        # print(samplearg)
        vel_x_measured_tmp_std.append(np.std(vel_x_measured[samplearg]))
    print(np.std(vel_x_measured_tmp_std))

    means.append(np.mean(vel_x_std))
    stds.append(np.std(vel_x_measured))
    plt.figure(figsize = (3, 3))
    plt.hist(std_errs_to_mean, bins = 40, density = True, rasterized=True)
    df = 2499
    mean, var, skew, kurt = t.stats(df, moments = 'mvsk')
    x = np.linspace(stats.t.ppf(0.01, df), stats.t.ppf(0.99, df), 100)
    plt.plot(x, stats.t.pdf(x, df), 'r-', lw = 3, alpha = 0.6, label = 't pdf')
    plt.xlabel("Studentised residual")
    plt.ylabel("Frequency of occurence")
    plt.tight_layout()
    plt.savefig(f"z-scores_{iterations}.pgf")
    plt.show()
    #plt.hist(vel_x_measured, bins = 50)
    #plt.show()
    #print(np.std(std_errs_to_mean))
    #print(np.mean(std_errs_to_mean))
    #zratio.append(np.std(std_errs_to_mean))
    print(stats.normaltest(std_errs_to_mean))
    print(stats.kurtosis(std_errs_to_mean))
    print(stats.kurtosistest(std_errs_to_mean))
    print(stats.kurtosistest(std_errs_to_mean, alternative = "two-sided"))
    print(stats.kurtosistest(std_errs_to_mean, alternative = "less"))
    print(stats.kurtosistest(std_errs_to_mean, alternative = "greater"))
    print(stats.skew(std_errs_to_mean))
    print(stats.skewtest(std_errs_to_mean))
    print(stats.skewtest(std_errs_to_mean, alternative = "two-sided"))
    print(stats.skewtest(std_errs_to_mean, alternative = "less"))
    print(stats.skewtest(std_errs_to_mean, alternative = "greater"))
    print("=====================")

    zratio.append(np.std(std_errs_to_mean))
    kurtosis.append(stats.kurtosis(std_errs_to_mean))
    skewness.append(stats.skew(std_errs_to_mean))


plt.scatter(its, zratio)
plt.xlabel("Iterations")
plt.ylabel("Z-score standard deviation")
plt.tight_layout()
plt.savefig("z-score.pgf")
plt.show()

plt.scatter(its, kurtosis)
plt.xlabel("Iterations")
plt.ylabel("Kurtosis of z-score")
plt.tight_layout()
plt.savefig("z-score-kurtosis.pgf")
plt.show()

plt.scatter(its, skewness)
plt.xlabel("Iterations")
plt.ylabel("Skewness of z-score")
plt.tight_layout()
plt.savefig("z-score-skewness.pgf")
plt.show()

plt.scatter(its, np.abs(np.array(means) - np.array(stds)))
plt.xlabel("Iterations")
plt.ylabel("Absolute difference between bootstrapped and non-bootstrapped standard errors")
plt.tight_layout()
plt.savefig("bsse-se-abs-diff.pgf")
plt.show()





"""vel_x_tmp = []
for _ in range(1000):
    piv.resample()
    piv.get_correlation_averaged_velocity_field()
    vel_x_tmp.append(piv.x_velocity_averaged().flatten()[0])
vel_x_std.append(np.std(vel_x_tmp))
print(np.std(vel_x_std))
print(np.std(vel_x_measured))
print(np.mean(vel_x_std))
plt.figure(figsize = (3.2, 2.6))
plt.hist(vel_x_tmp, bins = 50)
plt.xlabel("X Displacement (px)")
plt.ylabel("Frequency of Occurence")
plt.axvline(np.mean(vel_x_tmp) - np.std(vel_x_tmp), c = "orange", ls = ":")
plt.axvline(np.mean(vel_x_tmp) + np.std(vel_x_tmp), c = "orange", ls = ":")
#plt.savefig('piv_correlation_distribution.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()
"""