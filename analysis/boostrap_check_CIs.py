from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os
import math


def Gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / 2 * c**2)


def pdf(x, sigma, mean):
    return (1 / (sigma * (2 * math.pi)**(1/2))) * np.exp((-(x-mean)**2)/(2 * sigma**2))


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rc('text', usetex=True)
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.rcParams.update({
        "pgf.texsystem": "xelatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

folders = ["constant_vx3-0_vxsd1-0_vy3-0_vysd1-0_f2000", "gradient_vx3-25_vxsd1-0_vy0-0_vysd0-0_f500", "constant_vx0-0_vxsd0-0_vy3-0_vysd0-0_f500", "constant_vx3-0_vxsd0-0_vy0-0_vysd0-0_f500", "constant_vx3-0_vxsd0-0_vy3-0_vysd0-0_f500", "constant_vx3-5_vxsd0-0_vy3-5_vysd0-0_f500", "constant_vx3-25_vxsd0-0_vy3-25_vysd0-0_f500", "constant_vx3-25_vxsd1-0_vy3-25_vysd1-0_f500"]
folders = ["constant_vx3-25_vxsd0-0_vy3-25_vysd0-0_f500"]
samples = 500

for folder in folders:
    v_x = []
    v_y = []
    v_x_bs_mean = []
    v_y_bs_mean = []
    v_x_sd_bs = []
    v_y_sd_bs = []

    # Get mean measured x & y velocity vectors
    for filename in os.listdir(f"../data/simulated/{folder}"):
        piv = PIV(f"../data/simulated/{folder}/{filename}", 24, 15, 1, 0, "5pointgaussian", False)
        v_x.append(piv.correlation_averaged_velocity_field[0][0, 0, 2])
        v_y.append(piv.correlation_averaged_velocity_field[0][0, 0, 3])

        v_x_bs = []
        v_y_bs = []
        for _ in range(samples):
            piv.get_resampled_correlation_averaged_velocity_field()
            v_x_bs.append(piv.resampled_correlation_averaged_velocity_field[0][0, 0, 2])
            v_y_bs.append(piv.resampled_correlation_averaged_velocity_field[0][0, 0, 3])

        v_x_sd_bs.append(np.std(v_x_bs, ddof = 1))
        v_y_sd_bs.append(np.std(v_y_bs, ddof = 1))

    v_x_mean = np.mean(v_x)
    v_y_mean = np.mean(v_y)

    std_err_ratio = (v_x_mean - v_x) / v_x_sd_bs

    print(np.std(std_err_ratio))
    print(np.mean(std_err_ratio))

    plt.figure(figsize = (6.30045, 2.4))
    #plt.title(f"{folder}\n{np.std(std_err_ratio, ddof = 1)}")
    plt.hist(std_err_ratio, bins = 100)
    plt.xlabel("Standard errors from the mean")
    #xs = np.linspace(np.min(std_err_ratio), np.max(std_err_ratio), 1000)
    #ys = pdf(xs, np.std(std_err_ratio, ddof = 1), np.mean(std_err_ratio))
    #plt.plot(xs, ys, label = "Actual")
    #ys = pdf(xs, 1, np.mean(std_err_ratio))
    #plt.plot(xs, ys, label = "Expected")
    #ys = pdf(xs, 1, 0)
    #plt.plot(xs, ys, label = "True")
    #plt.legend()
    plt.tight_layout()
    plt.savefig(f"./visualisations/bootstrap_simulated/{folder}_ratio.pgf")
    plt.show()
