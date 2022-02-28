import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap

"""plt.rcParams["font.family"] = "serif"
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
    })"""
np.set_printoptions(threshold = np.inf)



numberofbins = 31
files = os.listdir("../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(0, 2 * np.pi, numberofbins)
indices = np.digitize(phases, bins)
vs = []
for i, b in enumerate(bins):
    if 1 == 1:
        filestopiv = np.array(files)[indices == i + 1]
        """print(f"Bin: {bins[23]}-{bins[24]}")
        print(f"minmax: {np.min(np.array(phases)[indices == i + 1])}-{np.max(np.array(phases)[indices == i + 1])}")"""
        if len(filestopiv) > 0:
            piv = PIV("", 24, 24, 16, 0.4, "5pointgaussian", False)
            piv.add_video(["../data/zebrafish/phase/" + str(f) for f in filestopiv])
            piv.set_coordinate(198, 234)
            #piv.get_spaced_coordinates()
            piv.get_correlation_matrices()
            piv.get_correlation_averaged_velocity_field()
            piv.get_velocity_field()
            plt.scatter(np.mean(np.array(phases)[indices == i + 1]), piv.x_velocity_averaged().flatten())
            plt.scatter(np.array(phases)[indices == i + 1], piv.x_velocity().flatten())
            #piv.plot_flow_field()
            """plt.hist(piv.x_velocity().flatten(), bins = len(filestopiv))
            plt.show()"""

            """U = piv.x_velocity_averaged()[:, :]
            V = piv.y_velocity_averaged()[:, :]
            mag = np.sqrt(U ** 2 + V ** 2)
            plt.figure(figsize = (3, 2.3))
            plt.imshow(piv.intensity_array_for_display, cmap = "gray", aspect = "auto", interpolation = "none")
            plt.quiver(piv.xcoords(), piv.ycoords(), U / mag, V / mag, mag, angles = "xy")
            plt.colorbar(label = "Displacement (px)")
            plt.axis('off')
            plt.savefig('../analysis/presentation/piv_zebrafish_flowfield.pgf', transparent = True)
            plt.show()"""

            """uncs = []
            for _ in range(10):
                uncertainty = piv.get_uncertainty(50)
                uncs.append(np.sqrt(uncertainty[0] ** 2 + uncertainty[1] ** 2))
            mean_uncs = np.mean(uncs, axis = 0)
            std_uncs = np.std(uncs, axis = 0)
            plt.imshow(mean_uncs)
            plt.show()
            peak = np.unravel_index(mean_uncs.argmax(), mean_uncs.shape, order = "C")
            print(np.max(mean_uncs))
            print(mean_uncs[peak])
            print(std_uncs[peak] / np.sqrt(10))"""

plt.show()

"""
piv = PIV(f"", 24, 24, 16, 0.5, "5pointgaussian", False)
piv.add_video(f"../data/zebrafish/processed/22.tif")
piv.get_spaced_coordinates()
piv.get_correlation_matrices()
piv.get_velocity_field()
piv.get_correlation_averaged_velocity_field()

piv.plot_flow_field()

U = piv.x_velocity_averaged()[:, :]
V = piv.y_velocity_averaged()[:, :]
mag = np.sqrt(U ** 2 + V ** 2)
plt.figure(figsize = (3, 2.3))
plt.imshow(piv.intensity_array_for_display, cmap = "gray", aspect = "auto", interpolation = "none")
plt.quiver(piv.xcoords(), piv.ycoords(), U / mag, V / mag, mag, angles = "xy")
plt.colorbar(label = "Displacement (px)")
plt.axis('off')
plt.savefig('../analysis/presentation/piv_zebrafish_flowfield.pgf', transparent = True)
plt.show()"""

"""uncertainty = piv.get_uncertainty(1000)
print(np.max(uncertainty[0]))
print(np.max(uncertainty[1]))
plt.imshow(uncertainty[0])
plt.colorbar()
plt.show()
plt.imshow(uncertainty[1])
plt.colorbar()
plt.show()"""

"""uncs = []
for _ in range(10):
    uncertainty = piv.get_uncertainty(50)
    uncs.append(np.sqrt(uncertainty[0]**2 + uncertainty[1]**2))
mean_uncs = np.mean(uncs, axis = 0)
std_uncs = np.std(uncs, axis = 0)
plt.imshow(mean_uncs)
plt.show()
peak = np.unravel_index(mean_uncs.argmax(), mean_uncs.shape, order = "C")
print(np.max(mean_uncs))
print(mean_uncs[peak])
print(std_uncs[peak] / np.sqrt(10))"""


"""def Gaussian(x, a, sigma, mean):
    return a * np.exp(- 0.5 * ((x - mean)**2 / (2 * sigma**2)))


plt.figure(figsize = (3, 2.3))
x = [0, 1, 2]
y = [50, 400, 300]
xs = np.linspace(-0.5, 2.5, 100)
fit = curve_fit(Gaussian, x, y)
plt.plot(xs, Gaussian(xs, *fit[0]), color = "crimson")
plt.scatter(x, y, zorder = 5)
plt.xlabel("Point")
plt.ylabel("Amplitude")
plt.xticks(x, ["$x_{i-1}$", "$x_{i}$", "$x_{i+1}$"])
plt.xlim(-0.5, 2.5)
plt.ylim(0, 500)
plt.tight_layout()
plt.axvline(fit[0][2], color = "gray", ls = ":")
plt.savefig('../analysis/presentation/3pointgaussianfit.pgf', transparent = True)
plt.show()"""

"""piv = PIV(f"", 24, 24, 1, 0, "5pointgaussian", False)
piv.add_video(f"../data/simulated/constant_for_presentation/0.tif")
piv.set_coordinate(36, 36)
piv.get_correlation_matrices()
piv.get_velocity_field()
piv.get_correlation_averaged_velocity_field()
#piv.plot_flow_field()


# Masking method: https://stackoverflow.com/questions/31877353/overlay-an-image-segmentation-with-numpy-and-matplotlib
mask = np.zeros(piv.video[0].shape)
mask[24:48, 24:48] = 1
masked = np.ma.masked_where(mask == 0, piv.video[0])
plt.figure(figsize = (2, 2.3))
plt.imshow(piv.video[0], alpha = 0.5, cmap = "gray")
plt.imshow(masked, interpolation = "none")
plt.axis('off')
plt.savefig('../analysis/presentation/piv_frame_a_highlighted.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

mask = np.zeros(piv.video[0].shape)
mask[0:24, 0:24] = 1
masked = np.ma.masked_where(mask == 0, piv.video[1])
plt.figure(figsize = (2, 2.3))
plt.imshow(piv.video[1], alpha = 0.5, cmap = "gray")
plt.imshow(masked, alpha = 1, interpolation = "none")
plt.arrow(36, 36, 12 - 36, 12 - 36, color = "red", head_width = 2, head_length = 2, length_includes_head = True)
plt.axis('off')
plt.savefig('../analysis/presentation/piv_frame_b_highlighted.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

mask = np.zeros(piv.video[0].shape)
mask[48:72, 48:72] = 1
masked = np.ma.masked_where(mask == 0, piv.video[1])
plt.figure(figsize = (2, 2.3))
plt.imshow(piv.video[1], alpha = 0.5, cmap = "gray")
plt.imshow(masked, alpha = 1, interpolation = "none")
plt.arrow(36, 36, 36 - 12, 36 - 12, color = "red", head_width = 2, head_length = 2, length_includes_head = True)
plt.axis('off')
plt.savefig('../analysis/presentation/piv_frame_b_highlighted_final.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

plt.figure(figsize = (2, 2.3))
plt.imshow(piv.video[0], interpolation = "none")
plt.axis('off')
plt.savefig('../analysis/presentation/piv_frame_a.pgf', bbox_inches='tight', pad_inches = 0, transparent=True)
plt.show()

plt.figure(figsize = (2, 2.3))
plt.imshow(piv.video[1], interpolation = "none")
plt.axis('off')
plt.savefig('../analysis/presentation/piv_frame_b.pgf', bbox_inches='tight', pad_inches = 0, transparent=True)
plt.show()

plt.figure(figsize = (2, 2.3))
x = np.flip(piv.correlation_matrices[0, 0, 0], axis = 1)
plt.imshow(x, origin = "lower", extent=[-x.shape[1]/2., x.shape[1]/2., -x.shape[0]/2., x.shape[0]/2], interpolation = "none")
plt.savefig('../analysis/presentation/piv_correlation.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

plt.figure(figsize = (2, 2.3))
x = np.flip(np.mean(piv.correlation_matrices, axis = 0)[0, 0], axis = 1)
plt.imshow(x, origin = "lower", extent=[-x.shape[1]/2., x.shape[1]/2., -x.shape[0]/2., x.shape[0]/2], interpolation = "none")
plt.savefig('../analysis/presentation/piv_correlation_averaged.pgf', bbox_inches='tight', pad_inches = 0, transparent = True)
plt.show()

print(piv.x_velocity()[0])
print(piv.y_velocity()[0])

print(piv.x_velocity_averaged())
print(piv.y_velocity_averaged())

print(piv.get_uncertainty(100))
"""