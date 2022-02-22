import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt

for filename in os.listdir(f"../data/zebrafish/processed/"):
    if filename == "22.tif":
        print(f"Processing: {filename}")
        piv = PIV(f"", 16, 24, 64, 0, "5pointgaussian", True)
        piv.add_video(f"../data/zebrafish/processed/{filename}")
        plt.imshow(piv.video[0])
        plt.show()
        plt.imshow(piv.video[0] / np.max(piv.video[0]) - piv.video[1] / np.max(piv.video[1]))
        plt.show()

        # First pass
        piv.get_spaced_coordinates()
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.window_deform()
        plt.imshow(piv.video[0])
        plt.show()
        plt.imshow(piv.video[0] / np.max(piv.video[0]) - piv.video[1] / np.max(piv.video[1]))
        plt.show()

        # Second pass
        piv.set(16, 6, 28)
        piv.get_spaced_coordinates()
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.window_deform()
        plt.imshow(piv.video[0])
        plt.show()
        plt.imshow(piv.video[0] / np.max(piv.video[0]) - piv.video[1] / np.max(piv.video[1]))
        plt.show()

        # Thid pass
        piv.set(8, 4, 16)
        piv.get_spaced_coordinates()
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.window_deform()
        plt.imshow(piv.video[0])
        plt.show()

        plt.imshow(piv.video[0] / np.max(piv.video[0]) - piv.video[1] / np.max(piv.video[1]))
        plt.show()

        # True
        plt.imshow(piv.video[1])
        plt.show()
