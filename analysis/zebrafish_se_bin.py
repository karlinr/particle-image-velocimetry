from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt

# MPL
# plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

xs = []
ys = []

piv = PIV(f"../data/zebrafish/processed/20.tif", 24, 24, 24, 0, "5pointgaussian", False)
piv.set_coordinate(201, 240)
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()
print(piv.x_velocity_averaged())
for s in range(piv.frames // 2 - 2):
    vels = []
    for i in range(1000):
        piv.resample(s + 2)
        piv.get_correlation_averaged_velocity_field()
        vels.append(piv.x_velocity_averaged()[0])
    xs.append(s + 2)
    ys.append(np.std(vels))

plt.scatter(xs, 1 / np.square(ys))
plt.xlabel("Number of correlation matrices")
plt.ylabel("$\sigma^{-2}$")
plt.show()
