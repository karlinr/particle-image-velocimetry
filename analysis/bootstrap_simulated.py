from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

folders = ["gradient_vx3-25_vxsd1-0_vy0-0_vysd0-0_f500", "constant_vx0-0_vxsd0-0_vy3-0_vysd0-0_f500", "constant_vx3-0_vxsd0-0_vy0-0_vysd0-0_f500", "constant_vx3-0_vxsd0-0_vy3-0_vysd0-0_f500", "constant_vx3-5_vxsd0-0_vy3-5_vysd0-0_f500", "constant_vx3-25_vxsd0-0_vy3-25_vysd0-0_f500", "constant_vx3-25_vxsd1-0_vy3-25_vysd1-0_f500"]
samples = 500

for folder in folders:
    v_x = []
    v_y = []
    v_x_sd_bs = []
    v_y_sd_bs = []

    # Get true distribution and bootstrapped distribution and save to arrays
    for filename in os.listdir(f"../data/simulated/{folder}"):
        print(filename)
        piv = PIV(f"../data/simulated/{folder}/{filename}", 24, 15, 1, 0, "9pointgaussian", False)
        v_x.append(piv.correlation_averaged_velocity_field[0][0, 0, 2])
        v_y.append(piv.correlation_averaged_velocity_field[0][0, 0, 3])

        v_x_bs = []
        v_y_bs = []

        # Get standard errors for bootstrap
        for _ in range(samples):
            piv.get_resampled_correlation_averaged_velocity_field()
            v_x_bs.append(piv.resampled_correlation_averaged_velocity_field[0][0, 0, 2])
            v_y_bs.append(piv.resampled_correlation_averaged_velocity_field[0][0, 0, 3])

        if not os.path.exists(f"./visualisations/bootstrap_simulated_distributions/{folder}"):
            os.makedirs(f"./visualisations/bootstrap_simulated_distributions/{folder}")
        plt.hist2d(v_x_bs, v_y_bs, bins = 100)
        plt.savefig(f"./visualisations/bootstrap_simulated_distributions/{folder}/{filename}.png")
        plt.close()

        v_x_sd_bs.append(np.std(v_x_bs))
        v_y_sd_bs.append(np.std(v_y_bs))

    # Get the standard error on the true distribution
    v_x_sd = np.std(v_x)
    v_y_sd = np.std(v_y)

    # Plot the true distribution
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12, 6))
    fig.suptitle(folder)
    ax1.hist(v_x, bins = 50)
    ax1.set_title("True distribution of x vector")
    ax2.hist(v_y, bins = 50)
    ax2.set_title("True distribution of y vector")
    ax3.hist(v_x_sd_bs, bins = 50, label = "Bootstrapped standard error")
    ax3.axvline(v_x_sd, color = "black", label = "Sample standard error")
    ax3.axvline(np.mean(v_x_sd_bs), color = "red", label = "Bootstrapped mean standard error")
    ax3.legend()
    ax3.set_title("Distribution of bootstrapped x standard error")
    ax4.hist(v_y_sd_bs, bins = 50, label = "Bootstrapped standard error")
    ax4.axvline(v_y_sd, color = "black", label = "Sample standard error")
    ax4.axvline(np.mean(v_y_sd_bs), color = "red", label = "Bootstrapped mean standard error")
    ax4.legend()
    ax4.set_title("Distribution of bootstrapped y standard error")
    plt.tight_layout()
    plt.savefig(f"./visualisations/bootstrap_simulated/{folder}")
    plt.show()

    plt.hist2d(v_x, v_y, bins = 100)
    plt.savefig(f"./visualisations/bootstrap_simulated_distributions/{folder}/00true.png")
    plt.close()
