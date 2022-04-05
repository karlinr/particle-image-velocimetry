import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp


def normalise(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


binsize = 31
files = os.listdir("../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(np.min(phases), np.max(phases), binsize)
indices = np.digitize(phases, bins)
filestopiv = np.array(files)[indices == 14 + 1]
phasestopiv = [float(os.path.splitext(filename)[0]) for filename in filestopiv]

piv = PIV(f"", 24, 24, 8, 0.4, "5pointgaussian", False)
piv.add_video(["../data/zebrafish/phase/" + str(f) for f in filestopiv])
piv.get_spaced_coordinates()
piv.get_correlation_matrices()
piv.get_correlation_averaged_velocity_field()

# Get unmodified video
frame_a = np.copy(piv.video[::2, int(np.min(piv.ycoords())):int(np.max(piv.ycoords())), int(np.min(piv.xcoords())): int(np.max(piv.xcoords()))])
frame_b = np.copy(piv.video[1::2, int(np.min(piv.ycoords())):int(np.max(piv.ycoords())), int(np.min(piv.xcoords())): int(np.max(piv.xcoords()))])

# Bootstrap uncertainties
std_x, std_y = piv.get_uncertainty(20)

piv.get_correlation_averaged_velocity_field()

# Apply upper stderr
piv.correlation_averaged_velocity_field[0, :, :, 0] += std_x
piv.correlation_averaged_velocity_field[0, :, :, 1] += std_y
piv.window_deform()
frame_a_upper = np.copy(piv.video[::2, int(np.min(piv.ycoords())):int(np.max(piv.ycoords())), int(np.min(piv.xcoords())): int(np.max(piv.xcoords()))])

# Reset frames
piv.video_reset()
piv.add_video(["../data/zebrafish/phase/" + str(f) for f in filestopiv])

# Apply lower stderr
piv.correlation_averaged_velocity_field[0, :, :, 0] -= std_x
piv.correlation_averaged_velocity_field[0, :, :, 1] -= std_y
piv.window_deform()
frame_a_lower = np.copy(piv.video[::2, int(np.min(piv.ycoords())):int(np.max(piv.ycoords())), int(np.min(piv.xcoords())): int(np.max(piv.xcoords()))])

for frame in range(len(frame_a)):
    plt.imshow(frame_a[frame])
    plt.savefig(f"../analysis/presentation/deformation/{phasestopiv[frame]}_a.png")
    plt.show()
    plt.imshow(frame_a_upper[frame])
    plt.savefig(f"../analysis/presentation/deformation/{phasestopiv[frame]}_a_lower.png")
    plt.show()
    plt.imshow(frame_a_lower[frame])
    plt.savefig(f"../analysis/presentation/deformation/{phasestopiv[frame]}_a_upper.png")
    plt.show()
    plt.imshow(frame_b[frame])
    plt.savefig(f"../analysis/presentation/deformation/{phasestopiv[frame]}_b.png")
    plt.show()
