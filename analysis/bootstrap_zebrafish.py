from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os



# Get bootstrap
for filename in os.listdir("../data/zebrafish/processed"):
    sds_x = []
    sds_y = []
    true_x = []
    true_y = []

    pivtest = PIV(f"../data/zebrafish/processed/{filename}", 24, 24, 24, 0.33, "5pointgaussian", True)

    samples = 100
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
                plt.xlabel(f"{np.std(vels_x[:, j, k].ravel()):.2f}")
                plt.ylabel(f"{np.std(vels_y[:, j, k].ravel()):.2f}")
                #plt.axvline(0, c = "black")
                plt.hist2d(vels_x[:, j, k].ravel(), vels_y[:, j, k].ravel(), bins = 100)
                #plt.axvline(pivtest.correlation_averaged_velocity_field[0][j, k, 2], c = "crimson")
                sds_x.append(np.std(vels_x[:, j, k]))
                sds_y.append(np.std(vels_y[:, j, k]))
                true_x.append(pivtest.correlation_averaged_velocity_field[0][j, k, 2])
                true_y.append(pivtest.correlation_averaged_velocity_field[0][j, k, 3])

    plt.tight_layout()
    #plt.savefig(f"visualisations/bootstrap_zebrafish_9pt/{filename}.png")
    plt.show()
    plt.close()

    plt.title(f"{filename}")
    plt.scatter(np.abs(true_x), sds_x)
    plt.show()
    plt.title(f"{filename}")
    plt.scatter(np.abs(true_y), sds_y)
    plt.show()
