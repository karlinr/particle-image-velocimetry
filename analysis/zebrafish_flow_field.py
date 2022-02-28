import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

piv = PIV(f"", 24, 24, 16, 0.4, "5pointgaussian", True)
piv.add_video("../data/zebrafish/processed/23.tif")
piv.get_spaced_coordinates()
#piv.add_video("../data/simulated/constant_for_presentation/0.tif")
#piv.set_coordinate(37, 37)
start_time = time.time()
piv.get_correlation_matrices()
print("--- %s seconds ---" % (time.time() - start_time))
piv.get_correlation_averaged_velocity_field()
piv.plot_flow_field()
plt.imshow(piv.video[0])
plt.axis('off')
plt.show()
piv.window_deform()
plt.imshow(piv.video[0])
plt.show()
plt.imshow(piv.video[1])
plt.show()