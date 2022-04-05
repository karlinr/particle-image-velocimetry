from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# Plots unbinned data + binned data for give binsizes
# Uses dot product to find vector similarity

# MPL
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rc('text', usetex=True)
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.rcParams.update({
        "pgf.texsystem": "xelatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

files = os.listdir("../data/zebrafish/phase/")

vs = []
phases = []
plt.figure(figsize = (8, 8))

binsize = 31
files = os.listdir("../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(np.min(phases), np.max(phases), binsize)
indices = np.digitize(phases, bins)

vs = []
vs_unb = []
phases = []
vs_ub_phases = []

filestopiv = np.array(files)[indices == 23 + 1]
for f in filestopiv:
    piv = PIV("", 24, 28, 24, 0.6, "5pointgaussian", False)
    piv.add_video(f"../data/zebrafish/phase/{f}")
    piv.set_coordinate(201, 240)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs_unb.append(piv.x_velocity_averaged()[0, 0])
    vs_ub_phases.append(float(os.path.splitext(f)[0]))

plt.hist(vs_unb)
plt.show()

vs_unb = np.array(vs_unb)
indices = np.where(vs_unb < 0)
print(indices)
print(filestopiv[indices])

piv = PIV("", 24, 28, 24, 0.6, "5pointgaussian", False)
piv.add_video(["../data/zebrafish/phase/" + str(f) for f in filestopiv[indices]])
piv.set_coordinate(201, 240)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
piv.get_velocity_field()
print(np.mean(piv.x_velocity().flatten()))
print(np.std(piv.x_velocity().flatten() / np.sqrt(len(filestopiv[indices]))))
print(piv.x_velocity_averaged())
print(piv.get_uncertainty(100))

files = os.listdir("../data/zebrafish/phase/")

vs = []
phases = []
plt.figure(figsize = (8, 8))

binsize = 31
files = os.listdir("../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(0, 2*np.pi, binsize)
indices = np.digitize(phases, bins)

vs = []
vs_unb = []
phases = []
vs_ub_phases = []

for i, b in enumerate(bins):
    if i == 23 + 1:
        print(bins[i + 1])
        print(bins[i + 2])
    filestopiv = np.array(files)[indices == i + 1]
    piv = PIV(b, 24, 28, 24, 0.6, "5pointgaussian", False)
    piv.add_video(["../data/zebrafish/phase/" + str(f) for f in filestopiv])
    if i == 23:
        plt.figure(figsize = (3, 2.3))
        plt.imshow(piv.intensity_array_for_display, cmap = "gray", aspect = "auto")
        rectangle = plt.Rectangle((234 - 12, 196 - 12), 24, 24, fc = 'none', ec = "red")
        plt.gca().add_patch(rectangle)
        plt.axis('off')
        plt.savefig('../analysis/presentation/analysis_region.pgf', transparent = True)
        plt.show()
    piv.set_coordinate(196, 234)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs.append(piv.x_velocity_averaged()[0, 0])
    v = [piv.x_velocity_averaged()[0, 0], piv.y_velocity_averaged()[0, 0]]
    #piv.plot_flow_field()
    #print(v)
    phases.append(b + (np.pi) / binsize)

    for f in filestopiv:
        piv = PIV(b, 64, 28, 24, 0.6, "5pointgaussian", False)
        piv.add_video(f"../data/zebrafish/phase/{f}")
        piv.set_coordinate(196, 234)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        vs_unb.append(piv.x_velocity_averaged()[0, 0])
        vs_ub_phases.append(float(os.path.splitext(f)[0]))

plt.figure(figsize = (3, 2.3))
plt.scatter(vs_ub_phases, vs_unb, s = 0.3, c = "black")
plt.scatter(phases, vs, c = "red", s = 2)
plt.xlabel("Phase (Rads)")
plt.ylabel("Displacement (px)")
plt.tight_layout()
#plt.savefig('../analysis/presentation/analysis_distribution.pgf', transparent = True)
plt.show()