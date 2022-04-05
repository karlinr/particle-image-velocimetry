import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap

"""plt.rcParams["font.family"] = "serif"
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
    })"""
np.set_printoptions(threshold = np.inf)



numberofbins = 31
files = os.listdir("../../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(0, 2 * np.pi, numberofbins)
indices = np.digitize(phases, bins)
vs = []
for i, b in enumerate(bins):
    if i == 23:
        filestopiv = np.array(files)[indices == i + 1]
        """print(f"Bin: {bins[23]}-{bins[24]}")
        print(f"minmax: {np.min(np.array(phases)[indices == i + 1])}-{np.max(np.array(phases)[indices == i + 1])}")"""
        if len(filestopiv) > 0:
            piv = PIV("", 40, 24, 16, 0.4, "5pointgaussian", False)
            piv.add_video(["../data/zebrafish/phase/" + str(f) for f in filestopiv])
            piv.set_coordinate(198, 235)
            #piv.set_coordinate(100, 150)
            #piv.get_spaced_coordinates()
            piv.get_correlation_matrices()
            piv.get_correlation_averaged_velocity_field()
            piv.plot_flow_field()
