from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

folders = ["constant_vx3-0_vxsd1-0_vy3-0_vysd1-0_f1", "constant_vx3-5_vxsd1-0_vy3-5_vysd1-0_f1", "constant_vx0-0_vxsd0-0_vy3-5_vysd0-0_f1", "constant_vx3-5_vxsd0-0_vy0-0_vysd0-0_f1", "constant_vx3-5_vxsd0-0_vy3-5_vysd0-0_f1"]
samples = 5000

for folder in folders:
    v_x = []
    v_y = []
    v_x_sd_bs = []
    v_y_sd_bs = []

    # Get true distribution and bootstrapped distribution and save to arrays
    for filename in os.listdir(f"../data/simulated/{folder}"):
        print(filename)
        piv = PIV(f"../data/simulated/{folder}/{filename}", 24, 15, 1, 0, "sinc", False)


        # Get standard errors for bootstrap
        for _ in range(samples):
            piv.get_resampled_correlation_averaged_velocity_field()
            v_x.append(piv.resampled_correlation_averaged_velocity_field[0][0, 0, 2])
            v_y.append(piv.resampled_correlation_averaged_velocity_field[0][0, 0, 3])

        if not os.path.exists(f"./visualisations/bootstrap_peak_locking/{folder}"):
            os.makedirs(f"./visualisations/bootstrap_peak_locking/{folder}")

        plt.title(f"{folder}")
        plt.imshow(piv.resampled_correlation_matrices_averaged[0][0,0,:,:])
        plt.savefig(f"./visualisations/bootstrap_peak_locking/{folder}/{filename}_sinc_corr.png")
        plt.show()

        plt.title(f"{folder}")
        plt.hist2d(v_x, v_y, bins = 100)
        plt.savefig(f"./visualisations/bootstrap_peak_locking/{folder}/{filename}_sinc.png")
        plt.show()
        plt.close()

for folder in folders:
    v_x = []
    v_y = []
    v_x_sd_bs = []
    v_y_sd_bs = []

    # Get true distribution and bootstrapped distribution and save to arrays
    for filename in os.listdir(f"../data/simulated/{folder}"):
        print(filename)
        piv = PIV(f"../data/simulated/{folder}/{filename}", 24, 15, 1, 0, "9pointgaussian", False)


        # Get standard errors for bootstrap
        for _ in range(samples):
            piv.get_resampled_correlation_averaged_velocity_field()
            v_x.append(piv.resampled_correlation_averaged_velocity_field[0][0, 0, 2])
            v_y.append(piv.resampled_correlation_averaged_velocity_field[0][0, 0, 3])

        if not os.path.exists(f"./visualisations/bootstrap_peak_locking/{folder}"):
            os.makedirs(f"./visualisations/bootstrap_peak_locking/{folder}")

        plt.title(f"{folder}")
        plt.imshow(piv.resampled_correlation_matrices_averaged[0][0,0,:,:])
        plt.savefig(f"./visualisations/bootstrap_peak_locking/{folder}/{filename}_9pt_corr.png")
        plt.show()

        plt.title(f"{folder}")
        plt.hist2d(v_x, v_y, bins = 100)
        plt.savefig(f"./visualisations/bootstrap_peak_locking/{folder}/{filename}_9pt.png")
        plt.show()
        plt.close()

for folder in folders:
    v_x = []
    v_y = []
    v_x_sd_bs = []
    v_y_sd_bs = []

    # Get true distribution and bootstrapped distribution and save to arrays
    for filename in os.listdir(f"../data/simulated/{folder}"):
        print(filename)
        piv = PIV(f"../data/simulated/{folder}/{filename}", 24, 15, 1, 0, "5pointgaussian", False)


        # Get standard errors for bootstrap
        for _ in range(samples):
            piv.get_resampled_correlation_averaged_velocity_field()
            v_x.append(piv.resampled_correlation_averaged_velocity_field[0][0, 0, 2])
            v_y.append(piv.resampled_correlation_averaged_velocity_field[0][0, 0, 3])

        if not os.path.exists(f"./visualisations/bootstrap_peak_locking/{folder}"):
            os.makedirs(f"./visualisations/bootstrap_peak_locking/{folder}")

        plt.title(f"{folder}")
        plt.imshow(piv.resampled_correlation_matrices_averaged[0][0,0,:,:])
        plt.savefig(f"./visualisations/bootstrap_peak_locking/{folder}/{filename}_5pt_corr.png")
        plt.show()

        plt.title(f"{folder}")
        plt.hist2d(v_x, v_y, bins = 100)
        plt.savefig(f"./visualisations/bootstrap_peak_locking/{folder}/{filename}_5pt.png")
        plt.close()
