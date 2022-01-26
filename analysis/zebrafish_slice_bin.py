from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, ifft


# MPL
#plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


files = sorted(os.listdir("../data/zebrafish/unbinned/"))

vs = []
phases = []
vs_binned = []
phases_binned = []

for filename in files:
    piv = PIV(f"../data/zebrafish/unbinned/{filename}", 24, 60, 24, 0, "5pointgaussian", False)
    piv.set_coordinate(201, 240)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs.append(piv.x_velocity_averaged()[0, 0])
    phases.append(float(os.path.splitext(filename)[0]))

for filename in os.listdir(f"../data/zebrafish/processed/"):
    piv = PIV(f"../data/zebrafish/processed/{filename}", 24, 60, 24, 0, "5pointgaussian", False)
    piv.set_coordinate(201, 240)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs_binned.append(piv.x_velocity_averaged()[0, 0])
    phases_binned.append(int(os.path.splitext(os.path.basename(filename))[0]) / 31 * (2 * np.pi) + (1/31 * (np.pi)))


plt.scatter(phases, vs, s = 1)
plt.scatter(phases_binned, vs_binned, s = 5, c = "black")
plt.show()

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