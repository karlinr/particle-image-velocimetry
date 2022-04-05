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
#plt.rc('text', usetex=True)
#plt.rcParams.update({"pgf.texsystem": "xelatex", "font.family": "serif", "text.usetex": True, 'pgf.rcfonts': False})


binsize = 31
files = os.listdir("../data/zebrafish/phasetime/")
phases = [float(os.path.splitext(filename)[0]) % (2 * np.pi) for filename in files]
times = [float(os.path.splitext(filename)[0]) / (2 * np.pi) for filename in files]
print(sorted(phases))
bins = np.linspace(0, 2 * np.pi, binsize)
print(bins)
indices = np.digitize(phases, bins)

for i, b in enumerate(bins):
    if i == 14:
        filestopiv = np.array(files)[indices == i + 1]
        print(np.array(phases)[indices == i + 1])
        print(np.min(np.array(phases)[indices == i + 1]))
        print(np.max(np.array(phases)[indices == i + 1]))
        piv = PIV(b, 64, 24, 1, 0.0, "9pointgaussian", False, True)
        piv.add_video(["../data/zebrafish/phasetime/" + str(f) for f in filestopiv])
        piv.set_coordinate(196, 234)
        #piv = PIV(f"", 52, 10, 1, 0, "9pointgaussian", False, True)
        #piv.add_video(f"../data/simulated/outliers/0.tif")
        #piv.set_coordinate(36, 36)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.get_velocity_field()
        print("first pass done")
        piv.do_pass()
        piv.set(24, 8, 1)
        #piv.set_coordinate(36, 36)
        piv.set_coordinate(196, 234)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.get_velocity_field()
        print("second pass done")

        true_x_velocity = piv.x_velocity_averaged().flatten()[0]
        print(true_x_velocity)
        true_x_velocity_unc = np.mean(piv.x_velocity().flatten())
        true_x_velocity_std = piv.get_uncertainty(1000)[0]

        print(piv.get_uncertainty(1000))
        print(np.std(piv.x_velocity().flatten()) / piv.x_velocity().flatten().shape[0] ** 0.5)

        unc_mean = []
        cor_mean = []
        phs_mean = []
        pek_mean = []
        tim_mean = []
        cor_dst_mean = []
        unc_dst_mean = []
        cor_prmsr_mean = []
        unc_prmsr_mean = []

        currentphases = np.array(phases)[indices == i + 1]
        currentimes = np.array(times)[indices == i + 1]

        print(piv.video.shape[0] // 4)

        for _ in range(10000):
            piv.resample(15)
            piv.get_correlation_averaged_velocity_field()
            unc_mean.append(np.mean(piv.x_velocity()[piv.samplearg].flatten()))
            cor_mean.append(piv.x_velocity_averaged().flatten()[0])
            phs_mean.append(np.mean(currentphases[piv.samplearg]))
            pek_mean.append(piv.get_averaged_peak_correlation_amplitude().flatten()[0])
            #cor_prmsr_mean.append(piv.get_averaged_peak_to_root_mean_square_ratio())
            #unc_prmsr_mean.append(np.mean(piv.get_peak_to_root_mean_square_ratio()))
            #cor_prmsr_mean.append(piv.get_averaged_peak_correlation_amplitude())
            #unc_prmsr_mean.append(np.mean(piv.get_peak_correlation_amplitude()))
            tim_mean.append(np.mean(currentimes[piv.samplearg]))

            unc_dst_mean.append(np.abs(np.mean(piv.x_velocity()[piv.samplearg].flatten()) - true_x_velocity_unc))
            cor_dst_mean.append(np.abs(piv.x_velocity_averaged().flatten()[0] - true_x_velocity))

        plt.title(i)
        plt.hist2d(unc_mean, cor_mean, bins = 100)
        plt.plot([-100, 100], [-100, 100], marker = "o", color = "red", alpha = 0.5)
        plt.axvline(true_x_velocity_unc, c = "red", alpha = 0.5)
        plt.axhline(true_x_velocity, c = "red", alpha = 0.5)
        plt.xlabel("Non-correlation averaged mean (px)")
        plt.ylabel("Correlation averaged mean (px)")
        plt.show()

        plt.title(i)
        plt.hist2d(phs_mean, cor_mean, bins = 100)
        plt.xlabel("Phase")
        plt.ylabel("Correlation averaged mean (px)")
        plt.axvline(np.mean(currentphases))
        plt.show()

        plt.title(i)
        plt.hist2d(tim_mean, cor_mean, bins = 100)
        plt.xlabel("Time")
        plt.ylabel("Correlation averaged mean (px)")
        plt.show()

        """plt.hist2d(unc_prmsr_mean, unc_dst_mean, bins = 100)
        plt.xlabel("PRMSR")
        plt.ylabel("Uncorrelation averaged residual")
        plt.show()

        plt.hist2d(cor_prmsr_mean, cor_dst_mean, bins = 100)
        plt.xlabel("PRMSR")
        plt.ylabel("Correlation averaged residual")
        plt.show()
        
        print(np.corrcoef(pek_mean, cor_mean))

        print(np.corrcoef([pek_mean, cor_mean, phs_mean, unc_mean]))"""


correlation = np.corrcoef([cor_mean, unc_mean, phs_mean, tim_mean])
names = ["Corr Mean", "Uncorr Mean", "Phase", "Time"]

fig, ax = plt.subplots()
im = ax.imshow(correlation, cmap = "RdBu")
im.set_clim(-1, 1)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(names)))
ax.set_xticklabels(names)
ax.set_yticks(np.arange(len(names)))
ax.set_yticklabels(names)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(correlation)):
    for j in range(len(correlation)):
        text = ax.text(j, i, f"{correlation[i, j]:.2f}",
                       ha="center", va="center", color="w")

ax.set_title("Correlations")
fig.tight_layout()
plt.show()
