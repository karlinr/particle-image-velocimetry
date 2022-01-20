from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Process zebrafish
for filename in os.listdir("../data/zebrafish/processed"):
    # Plot velocity field
    zebrafish = PIV(f"../data/zebrafish/processed/{filename}", 24, 24, 16, 0.6, "9pointgaussian", True)
    X = zebrafish.correlation_averaged_velocity_field[0][:, :, 0]
    Y = zebrafish.correlation_averaged_velocity_field[0][:, :, 1]
    U = zebrafish.correlation_averaged_velocity_field[0][:, :, 2]
    V = zebrafish.correlation_averaged_velocity_field[0][:, :, 3]
    mag = np.sqrt(U**2 + V**2)
    plt.figure(figsize = (12, 7))
    plt.title(f"{filename}")
    plt.imshow(np.flip(np.flip(np.rot90(zebrafish.intensity_array_for_display), axis = 1)), cmap = "gray", aspect = "auto")
    plt.quiver(X, Y, U / mag, V / mag, mag)
    plt.clim(0, 24)
    plt.colorbar()
    plt.xlim(0, zebrafish.intensity_array.shape[0])
    plt.ylim(0, zebrafish.intensity_array.shape[1])
    plt.savefig(f"visualisations/zebrafish_flow_field/{filename}.png")
    plt.show()
