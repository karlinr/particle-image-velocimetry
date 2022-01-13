from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os

# Process zebrafish
for filename in os.listdir("../data/simulated/constant_vx3-25_vxsd1-0_vy3-25_vysd1-0_f500/"):
    zebrafish = PIV(f"../data/simulated/constant_vx0-0_vxsd0-0_vy3-0_vysd0-0_f500/{filename}", 24, 15, 1, 0, "gaussian", False)
    """X = zebrafish.correlation_averaged_velocity_field[0][:, :, 0]
    Y = zebrafish.correlation_averaged_velocity_field[0][:, :, 1]
    U = zebrafish.correlation_averaged_velocity_field[0][:, :, 2]
    V = zebrafish.correlation_averaged_velocity_field[0][:, :, 3]
    mag = np.sqrt(U**2 + V**2)
    plt.imshow(np.flip(np.flip(np.rot90(zebrafish.intensity_array), axis = 1)), cmap = "Greys", aspect = "auto")
    plt.quiver(X, Y, U / mag, V / mag, mag)
    plt.clim(0, 10)
    plt.colorbar()
    plt.xlim(0, zebrafish.intensity_array.shape[0])
    plt.ylim(0, zebrafish.intensity_array.shape[1])
    plt.show()"""