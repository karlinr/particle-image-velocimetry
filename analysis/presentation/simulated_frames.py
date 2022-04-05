import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rc('text', usetex=True)
plt.rcParams.update({"pgf.texsystem": "xelatex", "font.family": "serif", "text.usetex": True, 'pgf.rcfonts': False})


piv = PIV(f"", 24, 24, 1, 0, "5pointgaussian", False)
piv.add_video(f"../../data/simulated/constant_for_presentation/0.tif")
piv.set_coordinate(36, 36)
piv.get_correlation_matrices()
piv.get_velocity_field()
piv.get_correlation_averaged_velocity_field()
print(piv.x_velocity_averaged())

fig, axs = plt.subplots(8, 8, figsize = (2.2, 2.2))
for i, cormat in enumerate(piv.correlation_matrices):
    axs[np.unravel_index(i, (8, 8))].imshow(cormat[0, 0], interpolation = "none")
    axs[np.unravel_index(i, (8, 8))].axis('off')
#plt.tight_layout()
plt.axis('off')
plt.savefig('piv_correlation_all.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

resample = np.random.choice(piv.video.shape[0] // 2, piv.video.shape[0] // 2)
fig, axs = plt.subplots(8,8, figsize = (2.2, 2.2))
for i, cormat in enumerate(piv.correlation_matrices[resample]):
    axs[np.unravel_index(i, (8, 8))].imshow(cormat[0, 0], interpolation = "none")
    axs[np.unravel_index(i, (8, 8))].axis('off')
#plt.tight_layout()
plt.axis('off')
plt.savefig('piv_correlation_all_resampled.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

plt.figure(figsize = (2.2, 2.2))
x = np.flip(np.mean(piv.correlation_matrices[resample], axis = 0)[0, 0], axis = 1)
plt.imshow(x, origin = "lower", extent=[-x.shape[1]/2., x.shape[1]/2., -x.shape[0]/2., x.shape[0]/2], interpolation = "none")
plt.axis('off')
plt.savefig('piv_correlation_averaged_resampled.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

# Masking method: https://stackoverflow.com/questions/31877353/overlay-an-image-segmentation-with-numpy-and-matplotlib
mask = np.zeros(piv.video[0].shape)
mask[24:48, 24:48] = 1
masked = np.ma.masked_where(mask == 0, piv.video[0])
plt.figure(figsize = (2, 2.3))
plt.imshow(piv.video[0], alpha = 0.5, cmap = "gray")
plt.imshow(masked, interpolation = "none")
plt.clim(np.min(piv.video[0]), np.max(piv.video[0]))
plt.axis('off')
plt.savefig('piv_frame_a_highlighted.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

mask = np.zeros(piv.video[0].shape)
mask[0:24, 0:24] = 1
masked = np.ma.masked_where(mask == 0, piv.video[1])
plt.figure(figsize = (2, 2.3))
plt.imshow(piv.video[1], alpha = 0.5, cmap = "gray")
plt.imshow(masked, alpha = 1, interpolation = "none")
plt.clim(np.min(piv.video[1]), np.max(piv.video[1]))
plt.arrow(36, 36, 12 - 36, 12 - 36, color = "red", head_width = 2, head_length = 2, length_includes_head = True)
plt.axis('off')
plt.savefig('piv_frame_b_highlighted.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

mask = np.zeros(piv.video[0].shape)
mask[48:72, 48:72] = 1
masked = np.ma.masked_where(mask == 0, piv.video[1])
plt.figure(figsize = (2, 2.3))
plt.imshow(piv.video[1], alpha = 0.5, cmap = "gray")
plt.imshow(masked, alpha = 1, interpolation = "none")
plt.clim(np.min(piv.video[1]), np.max(piv.video[1]))
plt.arrow(36, 36, 36 - 12, 36 - 12, color = "red", head_width = 2, head_length = 2, length_includes_head = True)
plt.axis('off')
plt.savefig('piv_frame_b_highlighted_final.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

plt.figure(figsize = (2, 2.3))
plt.imshow(piv.video[0], interpolation = "none")
plt.axis('off')
plt.savefig('piv_frame_a.pgf', bbox_inches='tight', pad_inches = 0, transparent=True)
plt.show()

plt.figure(figsize = (2, 2.3))
plt.imshow(piv.video[1], interpolation = "none")
plt.axis('off')
plt.savefig('piv_frame_b.pgf', bbox_inches='tight', pad_inches = 0, transparent=True)
plt.show()

plt.figure(figsize = (2, 2.3))
x = np.flip(piv.correlation_matrices[0, 0, 0], axis = 1)
plt.imshow(x, origin = "lower", extent=[-x.shape[1]/2., x.shape[1]/2., -x.shape[0]/2., x.shape[0]/2], interpolation = "none")
plt.axis('off')
plt.savefig('piv_correlation.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

plt.figure(figsize = (2.2, 2.2))
x = np.flip(np.mean(piv.correlation_matrices, axis = 0)[0, 0], axis = 1)
plt.imshow(x, origin = "lower", extent=[-x.shape[1]/2., x.shape[1]/2., -x.shape[0]/2., x.shape[0]/2], interpolation = "none")
plt.axis('off')
plt.savefig('piv_correlation_averaged.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

print(piv.x_velocity()[0])
print(piv.y_velocity()[0])

print(piv.x_velocity_averaged())
print(piv.y_velocity_averaged())

print(piv.get_uncertainty(100))

vs_single_x = []
vs_single_y = []
vs_multi_x = []
vs_multi_y = []
for video in os.listdir("../../data/simulated/constant_for_presentation/"):
    piv = PIV(f"", 24, 24, 1, 0, "5pointgaussian", False)
    piv.add_video(f"../../data/simulated/constant_for_presentation/{video}")
    piv.set_coordinate(36, 36)
    piv.get_correlation_matrices()
    piv.get_velocity_field()
    vs_single_x.append(piv.x_velocity().flatten())
    vs_single_y.append(piv.y_velocity().flatten())

    piv.get_correlation_averaged_velocity_field()
    vs_multi_x.append(piv.x_velocity_averaged().flatten())
    vs_multi_y.append(piv.y_velocity_averaged().flatten())
vs_single_x = np.array(vs_single_x).flatten()
vs_single_y = np.array(vs_single_y).flatten()
print(np.mean(vs_single_x))
print(np.mean(vs_single_y))
print(np.std(vs_single_x) / np.sqrt(len(vs_single_x)))
print(np.std(vs_single_y) / np.sqrt(len(vs_single_y)))

vs_multi_x = np.array(vs_multi_x).flatten()
vs_multi_y = np.array(vs_multi_y).flatten()
print(np.mean(vs_multi_x))
print(np.mean(vs_multi_y))
print(np.std(vs_multi_x) / np.sqrt(len(vs_multi_x)))
print(np.std(vs_multi_y) / np.sqrt(len(vs_multi_y)))