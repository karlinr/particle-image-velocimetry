from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

for filename in os.listdir(f"../data/zebrafish/processed/"):
    if filename == "5.tif":
        piv = PIV(f"", 24, 24, 24, 0, "9pointgaussian", False)
        piv.add_video(f"../data/zebrafish/processed/{filename}")
        #piv.set_coordinate(201, 240)
        piv.set_coordinate(50, 275)
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.plot_flow_field()
        mean_x = piv.x_velocity_averaged().flatten()[0]
        mean_y = piv.x_velocity_averaged().flatten()[0]
        mean_xy = [mean_x, mean_y]
        mean = (mean_x**2 + mean_y**2) ** 0.5

        vs_x = []
        vs_y = []
        vs = []
        vs_proj = []

        for i in range(5000):
            piv.resample()
            piv.get_correlation_averaged_velocity_field()
            vs_x.append(piv.x_velocity_averaged().flatten()[0])
            vs_y.append(piv.y_velocity_averaged().flatten()[0])
            vs.append((vs_x[-1]**2 + vs_x[-1]**2) ** 0.5)
            vs_proj.append(np.dot([vs_x[-1], vs_y[-1]], mean_xy) / mean)
        print(np.std(vs_x))

        plt.title(f"{filename} Bootstrapped Distribution")
        plt.hist(vs_proj, bins = 500)
        plt.xlabel("Projection onto mean (px)")
        plt.ylabel("Count")
        #plt.savefig(f"../analysis/visualisations/02022022/distribution/projection/{filename}")
        plt.show()

        plt.title(f"{filename} Bootstrapped Distribution")
        plt.hist(vs_x, bins = 500)
        plt.xlabel("x displacement (px)")
        #plt.savefig(f"../analysis/visualisations/02022022/distribution/x/{filename}")
        plt.show()

        plt.title(f"{filename} Bootstrapped Distribution")
        plt.hist(vs_y, bins = 500)
        plt.xlabel("y displacement (px)")
        #plt.savefig(f"../analysis/visualisations/02022022/distribution/y/{filename}")
        plt.show()
