from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# Plots unbinned data + binned data for give binsizes
# Uses dot product to find vector similarity

# MPL
#plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

files = os.listdir("../data/zebrafish/phase/")

vs = []
phases = []
plt.figure(figsize = (8, 8))

"""for filename in files:
    piv = PIV(f"../data/zebrafish/phase/{filename}", 24, 24, 24, 0, "5pointgaussian", False)
    piv.add_video(f"../data/zebrafish/phase/{filename}")
    piv.set_coordinate(201, 240)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs.append(piv.x_velocity_averaged()[0, 0])
    phases.append(float(os.path.splitext(filename)[0]))
plt.figure(figsize = (8, 8))
plt.scatter(phases, vs, s = 1, c = "black")"""
for binsize in [30]:
    files = os.listdir("../data/zebrafish/phase/")
    phases = [float(os.path.splitext(filename)[0]) for filename in files]
    bins = np.linspace(np.min(phases), np.max(phases), binsize)
    np.set_printoptions(threshold=np.inf)
    indices = np.digitize(phases, bins)

    vs = []
    vs_unb = []
    phases = []
    vs_ub_phases = []

    import time

    for i, b in enumerate(bins):
        filestopiv = np.array(files)[indices == i + 1]
        piv = PIV(b, 24, 24, 24, 0.6, "5pointgaussian", False)
        piv.add_video(["../data/zebrafish/phase/" + str(f) for f in filestopiv])
        piv.set_coordinate(201, 250)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        vs.append(np.sqrt(piv.x_velocity_averaged()[0, 0]**2 + piv.y_velocity_averaged()[0, 0]**2))
        v = [piv.x_velocity_averaged()[0, 0], piv.y_velocity_averaged()[0, 0]]
        #piv.plot_flow_field()
        #print(v)
        phases.append(b + (np.pi) / binsize)
        for f in filestopiv:
            piv = PIV(b, 24, 24, 24, 0.6, "5pointgaussian", False)
            piv.add_video(f"../data/zebrafish/phase/{f}")
            piv.set_coordinate(201, 240)
            piv.get_correlation_matrices()
            piv.get_correlation_averaged_velocity_field()
            #print([piv.x_velocity_averaged()[0, 0], piv.y_velocity_averaged()[0, 0]])
            vs_unb.append(np.dot([piv.x_velocity_averaged()[0, 0], piv.y_velocity_averaged()[0, 0]], v) / vs[-1])
            if(vs_unb[-1] > 500):
                print("")
                print(v)
                print([piv.x_velocity_averaged()[0, 0], piv.y_velocity_averaged()[0, 0]])
            vs_ub_phases.append(float(os.path.splitext(f)[0]))

    plt.scatter(vs_ub_phases, vs_unb, s = 1, c = "black")
    plt.scatter(phases, vs, label = f"Bins : {binsize}", c = "red")
plt.xlabel("Phase (Rads)")
plt.ylabel("Displacement (px)")
plt.legend()
plt.show()


vs = []
phases = []
vs_binned = []
phases_binned = []

"""for filename in files:
    piv = PIV(f"../data/zebrafish/unbinned/{filename}", 24, 24, 24, 0, "5pointgaussian", False)
    piv.add_video(f"../data/zebrafish/unbinned/{filename}")
    piv.set_coordinate(201, 240)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs.append(piv.x_velocity_averaged()[0, 0])
    phases.append(float(os.path.splitext(filename)[0]))
    piv.plot_flow_field()

for filename in os.listdir(f"../data/zebrafish/processed/"):
    piv = PIV(f"../data/zebrafish/processed/{filename}", 24, 24, 24, 0, "5pointgaussian", False)
    piv.add_video(f"../data/zebrafish/processed/{filename}")
    piv.set_coordinate(201, 240)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs_binned.append(piv.x_velocity_averaged()[0, 0])
    phases_binned.append(int(os.path.splitext(os.path.basename(filename))[0]) / 31 * (2 * np.pi) + (1/31 * (np.pi)))


plt.scatter(phases, vs, s = 1)
plt.scatter(phases_binned, vs_binned, s = 5, c = "black")
plt.show()"""

"""# Setup Arrays
phases = []
vels = []
stderrs = []
allvels = []
allvelsbs = []
allvels_phase = []
allvelsbs_phase = []

piv = PIV(f"../data/zebrafish/unprocessed/030298.tif", 24, 24, 24, 0, "5pointgaussian", False)
piv.set_coordinate(201, 240)
piv.get_correlation_matrices()
piv.get_velocity_field()
xs = piv.x_velocity().flatten()
plt.scatter(range(len(xs)), xs, s = 2)
plt.show()

X = fft(xs)
N = len(X)
n = np.arange(N)
T = N/0.0001
freq = n/T
plt.plot(freq, abs(X))
plt.show()"""