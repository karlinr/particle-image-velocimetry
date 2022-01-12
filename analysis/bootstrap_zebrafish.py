from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# Get bootstrap
for filename in os.listdir("../data/zebrafish/processed"):
    pivtest = PIV(f"../data/zebrafish/processed/{filename}", 24, 15, 24, 0.26, "9pointgaussian", True)

    samples = 5000
    vels_x = np.empty((samples, pivtest.width, pivtest.height), dtype = np.float64)
    vels_y = np.empty((samples, pivtest.width, pivtest.height), dtype = np.float64)

    for i in range(samples):
        pivtest.get_resampled_correlation_averaged_velocity_field()
        vels_x[i, :, :] = pivtest.resampled_correlation_averaged_velocity_field[0][:, :, 2]
        vels_y[i, :, :] = pivtest.resampled_correlation_averaged_velocity_field[0][:, :, 3]

    # Plot histograms
    fig, ax = plt.subplots(figsize = (32, 26), sharex = True)
    plt.axis('off')

    for j in range(0, pivtest.width):
        for k in range(0, pivtest.height):
            if pivtest.threshold_array[j, k]:
                plt.title(f"{j}, {k}")
                fig.add_subplot(pivtest.height, pivtest.width, k * pivtest.width + j + 1)
                #plt.axvline(0, c = "black")
                plt.hist2d(vels_x[:, j, k].ravel(), vels_y[:, j, k].ravel(), bins = 100)
                #plt.axvline(pivtest.correlation_averaged_velocity_field[0][j, k, 2], c = "crimson")
    plt.tight_layout()
    plt.savefig(f"visualisations/bootstrap_zebrafish/{filename}.png")
    plt.show()