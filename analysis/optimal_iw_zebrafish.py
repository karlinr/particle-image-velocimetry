from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

iws = range(1, 40)
samples = 300

zebrafish = PIV(f"../data/zebrafish/processed/22.tif", 24, 24, 32, 0.5, "5pointgaussian", True)
sds_x = np.empty((len(iws), zebrafish.width, zebrafish.height), dtype = np.float64)
sds_y = np.empty((len(iws), zebrafish.width, zebrafish.height), dtype = np.float64)

# Process zebrafish
for iw in iws:
    zebrafish = PIV(f"../data/zebrafish/processed/22.tif", iw, 24, 32, 0.5, "5pointgaussian", True)

    vels_x = np.empty((samples, zebrafish.width, zebrafish.height), dtype = np.float64)
    vels_y = np.empty((samples, zebrafish.width, zebrafish.height), dtype = np.float64)

    for i in range(samples):
        zebrafish.get_resampled_correlation_averaged_velocity_field()
        vels_x[i, :, :] = zebrafish.resampled_correlation_averaged_velocity_field[0][:, :, 2]
        vels_y[i, :, :] = zebrafish.resampled_correlation_averaged_velocity_field[0][:, :, 3]

    sds_x[iw - 1, :, :] = np.std(vels_x, axis = 0)
    sds_y[iw - 1, :, :] = np.std(vels_y, axis = 0)

plt.axis('off')
fig, ax = plt.subplots(figsize = (32, 26), sharex = True)
for j in range(0, zebrafish.width):
    for k in range(0, zebrafish.height):
        if zebrafish.threshold_array[j, k]:
            plt.title(f"{j}, {k}")
            fig.add_subplot(zebrafish.height, zebrafish.width, k * zebrafish.width + j + 1)
            plt.scatter(iws, sds_x[:, j, k], s = 6)

plt.tight_layout()
plt.show()
plt.close()

plt.axis('off')
fig, ax = plt.subplots(figsize = (32, 26), sharex = True)
for j in range(0, zebrafish.width):
    for k in range(0, zebrafish.height):
        if zebrafish.threshold_array[j, k]:
            plt.title(f"{j}, {k}")
            fig.add_subplot(zebrafish.height, zebrafish.width, k * zebrafish.width + j + 1)
            plt.scatter(iws, sds_y[:, j, k], s = 6)

plt.tight_layout()
plt.show()
plt.close()
