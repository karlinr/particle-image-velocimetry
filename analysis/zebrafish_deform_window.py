import PIL.ImageTransform
import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import trackpy as tp
import tifffile as tf

if isinstance(a, np.ndarray):
    print("TRUE")

for filename in os.listdir(f"../data/zebrafish/processed/"):
    print(f"Processing: {filename}")
    # Load processed video
    piv = PIV(f"", 24, 24, 24, 0, "5pointgaussian", True)
    piv.add_video(f"../data/zebrafish/processed/{filename}")
    piv.get_spaced_coordinates()
    piv.get_correlation_matrices()
    piv.get_correlation_averaged_velocity_field()
    piv.plot_flow_field()

    # Get coords and velocities
    xcoords = piv.xcoords().astype(int)
    ycoords = piv.ycoords().astype(int)
    flow_x = -piv.x_velocity_averaged()
    flow_y = -piv.y_velocity_averaged()

    # Create transformation mesh
    mesh = []
    for i in range(xcoords.shape[0] - 1):
        for j in range(ycoords.shape[1] - 1):
            mesh.append([(xcoords[i, j], ycoords[i, j], xcoords[i + 1, j + 1], ycoords[i + 1, j + 1]),
                         [xcoords[i, j] + flow_x[i, j], ycoords[i, j] + flow_y[i, j],
                          xcoords[i, j + 1] + flow_x[i, j + 1], ycoords[i, j + 1] + flow_y[i, j + 1],
                          xcoords[i + 1, j + 1] + flow_x[i + 1, j + 1], ycoords[i + 1, j + 1] + flow_y[i + 1, j + 1],
                          xcoords[i + 1, j] + flow_x[i + 1, j], ycoords[i + 1, j] + flow_y[i + 1, j]]])


    for frame in range(piv.video.shape[0] // 2):
        image1 = np.array(np.flip(np.flip(np.rot90(piv.video[2 * frame, :, :]), axis = 1)))
        image2 = np.array(np.flip(np.flip(np.rot90(piv.video[2 * frame + 1, :, :]), axis = 1)))
        imagedist = Image.fromarray(image1)
        imagedist = np.array(imagedist.transform(imagedist.size, PIL.Image.MESH, mesh))


        def normalise(arr):
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


        image1 = normalise(image1)
        image2 = normalise(image2)
        imagedist = normalise(imagedist)

        image1 = image1[np.min(ycoords): np.max(ycoords), np.min(xcoords): np.max(xcoords)]
        image2 = image2[np.min(ycoords): np.max(ycoords), np.min(xcoords): np.max(xcoords)]
        imagedist = imagedist[np.min(ycoords): np.max(ycoords), np.min(xcoords): np.max(xcoords)]

        # fig, axs = plt.subplots(2, 2, figsize = (9, 9))
        # Load frame 1 and plot
        # axs[0, 0].set_title(f"Frame: {frame}")
        # axs[0, 0].imshow(np.array(image1))
        # Load frame 2 and plot
        # axs[0, 1].set_title(f"Frame: {frame + 1}")
        # axs[0, 1].imshow(np.array(image2))
        # Load frame 1, distort and plot

        # axs[1, 0].set_title("Distorted")
        # axs[1, 0].imshow(np.array(imagedist))
        # Plot difference
        # axs[1, 1].set_title("Difference")
        # im = axs[1, 1].imshow(np.array(np.flip(np.flip(np.rot90(piv.video[2 * frame + 1, :, :]) / np.max(np.array(piv.video[2 * frame + 1, :, :]))), axis = 1)) - np.array(imagedist) / np.max(np.array(imagedist)), vmin = -1, vmax = 1)
        # plt.colorbar(im, ax = axs[1, 1])
        # plt.show()

        if not os.path.exists(f"../analysis/visualisations/image_transform2/{os.path.splitext(filename)[0]}"):
            os.makedirs(f"../analysis/visualisations/image_transform2/{os.path.splitext(filename)[0]}")

        """plt.imshow((image1 - imagefor) + (image2 - imagebac))
        plt.colorbar()
        plt.show()"""

        plt.imshow(image2 * imagedist - image2 * np.max(image2 * imagedist))
        plt.colorbar()
        plt.show()
        plt.title(f"framepair:{frame}, frame:a")
        plt.imshow(image1, interpolation = "none")
        plt.savefig(f"../analysis/visualisations/image_transform2/{os.path.splitext(filename)[0]}/{frame}_a.png")
        plt.show()
        plt.close()
        plt.title(f"framepair:{frame}, frame:b")
        plt.imshow(image2, interpolation = "none")
        plt.savefig(f"../analysis/visualisations/image_transform2/{os.path.splitext(filename)[0]}/{frame}_b.png")
        plt.show()
        plt.close()
        plt.title(f"framepair:{frame}, Deformation a=>b")
        plt.imshow(imagedist, interpolation = "none")
        plt.savefig(f"../analysis/visualisations/image_transform2/{os.path.splitext(filename)[0]}/{frame}_c.png")
        plt.show()
        plt.close()
        plt.title(f"framepair:{frame}, Difference")
        plt.imshow(image2 - imagedist, interpolation = "none")
        plt.colorbar()
        plt.savefig(f"../analysis/visualisations/image_transform2/{os.path.splitext(filename)[0]}/{frame}_d.png")
        plt.show()
        plt.close()

        """image1 = image1[int(piv.sa + 0.5 * piv.iw):int(image1.shape[0] - 2 * piv.sa + piv.iw), int(piv.sa + 0.5 * piv.iw):int(image1.shape[1] - 2 * piv.sa + piv.iw)]
        image2 = image2[int(piv.sa + 0.5 * piv.iw):int(image2.shape[0] - 2 * piv.sa + piv.iw), int(piv.sa + 0.5 * piv.iw):int(image2.shape[1] - 2 * piv.sa + piv.iw)]
        imagedist = imagedist[int(piv.sa + 0.5 * piv.iw):int(imagedist.shape[0] - 2 * piv.sa + piv.iw), int(piv.sa + 0.5 * piv.iw):int(imagedist.shape[1] - 2 * piv.sa + piv.iw)]


        def normalise(arr):
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        image1 = normalise(image1)
        image2 = normalise(image2)
        imagedist = normalise(imagedist)

        f = tp.locate(image1, 19, invert = False, minmass = 1, threshold = 1/255)
        tp.annotate(f, image1)

        tp.subpx_bias(f)

        f = tp.locate(image2, 19, invert = False, minmass = 1, threshold = 1/255)
        tp.annotate(f, image2)
        f = tp.locate(imagedist, 19, invert = False, minmass = 1, threshold = 1/255)
        tp.annotate(f, imagedist)"""
