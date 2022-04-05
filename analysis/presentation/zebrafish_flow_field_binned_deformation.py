import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import skew, kurtosis

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rc('text', usetex=True)
plt.rcParams.update({"pgf.texsystem": "xelatex", "font.family": "serif", "text.usetex": True, 'pgf.rcfonts': False})


binsize = 31
files = os.listdir("../../data/zebrafish/phase/")
phases = [float(os.path.splitext(filename)[0]) for filename in files]
bins = np.linspace(np.min(phases), np.max(phases), binsize)
indices = np.digitize(phases, bins)

for i, b in enumerate(bins):
    if i == 14:
        filestopiv = np.array(files)[indices == i + 1]
        piv = PIV(b, 24, 24, 16, 0.0, "9pointgaussian", False)
        piv.add_video(["../../data/zebrafish/phase/" + str(f) for f in filestopiv])
        piv.get_spaced_coordinates()
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.get_velocity_field()

        video_frame1 = []
        video_frame1_transformed = []
        video_frame2 = []


        for indice in range(piv.video.shape[0] // 2):
            video_frame1.append(np.copy(piv.video[indice * 2, int(np.min(piv.ycoords())): int(np.max(piv.ycoords())), int(np.min(piv.xcoords())): int(np.max(piv.xcoords()))]))
        piv.window_deform()

        for indice in range(piv.video.shape[0] // 2):
            video_frame1_transformed.append(np.copy(piv.video[indice * 2, int(np.min(piv.ycoords())): int(np.max(piv.ycoords())), int(np.min(piv.xcoords())): int(np.max(piv.xcoords()))]))
            video_frame2.append(np.copy(piv.video[indice * 2 + 1, int(np.min(piv.ycoords())): int(np.max(piv.ycoords())), int(np.min(piv.xcoords())): int(np.max(piv.xcoords()))]))

        for indice in range(piv.video.shape[0] // 2):
            if indice == 0:
                print(indice)
                plt.figure(figsize = (3.2, 2.6))
                plt.imshow(video_frame1[indice])
                plt.axis('off')
                plt.savefig('transform_a.pgf', transparent = True)
                plt.show()
                plt.figure(figsize = (3.2, 2.6))
                plt.imshow(video_frame1_transformed[indice])
                plt.axis('off')
                plt.savefig('transform_a_transformed.pgf', transparent = True)
                plt.show()
                plt.figure(figsize = (3.2, 2.6))
                plt.imshow(video_frame2[indice])
                plt.axis('off')
                plt.savefig('transform_b.pgf', transparent = True)
                plt.show()

