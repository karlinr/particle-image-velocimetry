from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# MPL
#plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

vs = []
ts = []

files = os.listdir("../../data/zebrafish/time/")
print(files)

for i, filename in enumerate(os.listdir("../../data/zebrafish/time/")):
    if i % 32 == 0:
        piv = PIV(f"../data/zebrafish/time/{filename}", 24, 24, 24, 0, "5pointgaussian", False)
        piv.add_video(f"../data/zebrafish/time/{filename}")
    else:
        print(i)
        piv.add_video(f"../data/zebrafish/time/{filename}")
        piv.set_coordinate(201, 240)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        vs.append(piv.x_velocity_averaged().flatten()[0])
        if piv.x_velocity_averaged().flatten()[0] < -10:
            print(filename)
            plt.imshow(piv.correlation_averaged[0][0, 0, :, :])
            plt.show()
        ts.append(float(os.path.splitext(os.path.basename(filename))[0]))


plt.scatter(ts, vs, s = 1)
plt.show()
