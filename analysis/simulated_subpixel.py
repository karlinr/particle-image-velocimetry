from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# MPL
#plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

vs = []
piv = PIV(f"../data/simulated/constant3.5/1.tif", 24, 24, 24, 0, "5pointgaussian", False)
piv.set_coordinate(36, 36)
piv.get_correlation_matrices()
for i in range(5000):
    piv.resample()
    piv.get_correlation_averaged_velocity_field()
    vs.append(piv.x_velocity_averaged().flatten()[0])
print(np.mean(vs))
print(np.std(vs, ddof = 1))
plt.hist(vs, bins = 100)
plt.show()

vs = []
for filename in os.listdir("../data/simulated/constant3.5/"):
    piv = PIV(f"../data/simulated/constant3.5/{filename}", 24, 24, 24, 0, "5pointgaussian", False)
    piv.set_coordinate(36, 36)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs.append(piv.x_velocity_averaged().flatten()[0])

print(np.mean(vs))
print(np.std(vs, ddof = 1))
plt.hist(vs, bins = 100)
plt.show()

"""vs = []
for filename in os.listdir("../data/simulated/constant3.25/"):
    piv = PIV(f"../data/simulated/constant3.25/{filename}", 24, 24, 24, 0, "5pointgaussian", False)
    piv.set_coordinate(40, 40)
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    vs.append(piv.x_velocity_averaged().flatten()[0])

print(np.mean(vs))
plt.hist(vs, bins = 100)
plt.show()
"""