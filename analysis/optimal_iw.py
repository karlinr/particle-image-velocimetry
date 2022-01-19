from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

iws = range(1, 40)

stderrs = []

for iw in iws:
    print(iw)
    v_x = []
    v_y = []
    for filename in os.listdir("../data/simulated/iw_investigation"):
        piv = PIV(f"../data/simulated/iw_investigation/{filename}", iw, 20, np.max(iws) + iw, 0, "5pointgaussian", False)
        v_x.append(piv.correlation_averaged_velocity_field[0][0, 0, 2])
        v_y.append(piv.correlation_averaged_velocity_field[0][0, 0, 3])
    stderrs.append(np.std(v_x))
    """plt.title(f"Interrogation window: {iw}; Standard Error: {np.std(v_x):.2f}")
    plt.hist2d(v_x, v_y, bins = 50)
    plt.show()"""

plt.loglog(iws, stderrs)
plt.show()