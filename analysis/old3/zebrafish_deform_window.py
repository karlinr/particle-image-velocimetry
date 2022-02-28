import os
from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp

def normalise(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

for filename in os.listdir(f"../../data/zebrafish/processed/"):
    if filename == "11.tif":

        print(f"Processing: {filename}")
        # Do PIV
        piv = PIV(f"", 24, 24, 8, 0.4, "5pointgaussian", True)
        piv.add_video(f"../data/zebrafish/processed/{filename}")
        piv.get_spaced_coordinates()
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.plot_flow_field()

        # Get unmodified video
        frame_a = np.copy(piv.video[::2, int(np.min(piv.xcoords())):int(np.max(piv.xcoords())), int(np.min(piv.ycoords())): int(np.max(piv.ycoords()))])
        frame_b = np.copy(piv.video[1::2, int(np.min(piv.xcoords())):int(np.max(piv.xcoords())), int(np.min(piv.ycoords())): int(np.max(piv.ycoords()))])

        # Bootstrap uncertainties
        std_x, std_y = piv.get_uncertainty(20)

        piv.get_correlation_averaged_velocity_field()

        # Apply upper stderr
        piv.correlation_averaged_velocity_field[0, :, :, 2] += std_x
        piv.correlation_averaged_velocity_field[0, :, :, 3] += std_y
        piv.window_deform()
        print(np.min(piv.ycoords()))
        print(np.min(piv.xcoords()))
        frame_a_upper = np.copy(piv.video[::2, int(np.min(piv.xcoords())):int(np.max(piv.xcoords())), int(np.min(piv.ycoords())): int(np.max(piv.ycoords()))])

        # Reset frames
        piv.video_reset()
        piv.add_video(f"../data/zebrafish/processed/{filename}")

        # Apply lower stderr
        piv.correlation_averaged_velocity_field[0, :, :, 2] -= std_x
        piv.correlation_averaged_velocity_field[0, :, :, 3] -= std_y
        piv.window_deform()
        frame_a_lower = np.copy(piv.video[::2, int(np.min(piv.xcoords())):int(np.max(piv.xcoords())), int(np.min(piv.ycoords())): int(np.max(piv.ycoords()))])

        for frame in range(len(frame_a)):
            f = frame
            if f == 14:
                plt.imshow(frame_a[f])
                plt.show()
                plt.imshow(frame_a_upper[f])
                plt.show()
                plt.imshow(frame_a_lower[f])
                plt.show()
                plt.imshow(frame_b[f])
                plt.show()
                plt.imshow(2 * (frame_b[f].astype(np.intc)) - frame_a_upper[f].astype(np.intc) - frame_a_lower[f].astype(np.intc))
                plt.show()

            print(np.sum(2 * (frame_b.astype(np.intc)) - frame_a_upper.astype(np.intc) - frame_a_lower.astype(np.intc)))

            #plt.imshow(normalise(frame_a_upper[f]) + normalise(frame_a_lower[f]) * normalise(frame_b[f]) - 2*normalise(frame_b[f]**2))
            """plt.imshow(normalise(frame_a_lower[f]) + normalise(frame_b[f]) - 2 * normalise(frame_b[f]))
            plt.show()"""

        # Second pass
        """print("Second pass")
        piv.set(16, 16, 16)
        piv.get_spaced_coordinates()
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.window_deform(0.5)
        pass_current.append(np.copy(piv.video))
        piv.plot_flow_field()
    
        print("Third pass")
        piv.set(16, 8, 8)
        piv.get_spaced_coordinates()
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.window_deform(1)
        pass_current.append(np.copy(piv.video))
        piv.plot_flow_field()"""

        """# Third pass
        print("Third pass")
        piv.set(16, 6, 6)
        piv.get_spaced_coordinates()
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.window_deform()
        pass_current.append(np.copy(piv.video))
        piv.plot_flow_field()
    
        # fourth pass
        print("Fourth pass")
        piv.set(16, 4, 4)
        piv.get_spaced_coordinates()
        piv.get_correlation_matrices()
        piv.get_correlation_averaged_velocity_field()
        piv.window_deform()
        pass_current.append(np.copy(piv.video))
        piv.plot_flow_field()"""

        if not os.path.exists(f"../analysis/visualisations/window_deformation/differences/{os.path.splitext(filename)[0]}"):
            os.makedirs(f"../analysis/visualisations/window_deformation/differences/{os.path.splitext(filename)[0]}")
        if not os.path.exists(f"../analysis/visualisations/window_deformation/deformations/{os.path.splitext(filename)[0]}"):
            os.makedirs(f"../analysis/visualisations/window_deformation/deformations/{os.path.splitext(filename)[0]}")

        for frame in range(piv.video.shape[0] // 2):
            print(f"saving {frame}")
            """fig, axs = plt.subplots(1, 5, figsize = (16, 4))
            plt.suptitle(f"{frame} - Difference Between Frame b and Morphed Frame a")
            axs[0].set_title("Initial")
            axs[0].imshow(pass_current[0][frame * 2] / np.max(pass_current[0][frame * 2]) - piv.video[frame * 2 + 1] / np.max(piv.video[frame * 2 + 1]))
            axs[1].set_title("First Pass")
            axs[1].imshow(pass_current[1][frame * 2] / np.max(pass_current[1][frame * 2]) - piv.video[frame * 2 + 1] / np.max(piv.video[frame * 2 + 1]))
            axs[2].set_title("Second Pass")
            axs[2].imshow(pass_current[2][frame * 2] / np.max(pass_current[2][frame * 2]) - piv.video[frame * 2 + 1] / np.max(piv.video[frame * 2 + 1]))
            axs[3].set_title("Third Pass")
            axs[3].imshow(pass_current[3][frame * 2] / np.max(pass_current[3][frame * 2]) - piv.video[frame * 2 + 1] / np.max(piv.video[frame * 2 + 1]))
            axs[4].set_title("Fourth Pass")
            axs[4].imshow(pass_current[4][frame * 2] / np.max(pass_current[3][frame * 2]) - piv.video[frame * 2 + 1] / np.max(piv.video[frame * 2 + 1]))
            plt.tight_layout()
            plt.savefig(f"../analysis/visualisations/window_deformation/differences/{os.path.splitext(filename)[0]}/{frame}.png")
            plt.close()"""

            plt.figure(figsize = (12, 9))
            plt.title(f"{frame} Initial")
            plt.imshow(pass_current[0][frame * 2])
            plt.savefig(f"../analysis/visualisations/window_deformation/deformations/{os.path.splitext(filename)[0]}/{frame}_pass0.png")
            plt.close()
            """plt.figure(figsize = (12, 9))
            plt.title(f"{frame} pass 1")
            plt.imshow(pass_current[1][frame * 2])
            plt.savefig(f"../analysis/visualisations/window_deformation/deformations/{os.path.splitext(filename)[0]}/{frame}_pass1.png")
            plt.close()
            plt.figure(figsize = (12, 9))
            plt.title(f"{frame} pass 2")
            plt.imshow(pass_current[2][frame * 2])
            plt.savefig(f"../analysis/visualisations/window_deformation/deformations/{os.path.splitext(filename)[0]}/{frame}_pass2.png")
            plt.close()"""
            plt.figure(figsize = (12, 9))
            plt.title(f"{frame} pass {len(pass_current)}")
            plt.imshow(pass_current[-1][frame * 2])
            plt.savefig(f"../analysis/visualisations/window_deformation/deformations/{os.path.splitext(filename)[0]}/{frame}_pass1.png")
            plt.close()
            plt.figure(figsize = (12, 9))
            plt.title(f"{frame} Actual")
            plt.imshow(pass_current[0][frame * 2 + 1])
            plt.savefig(f"../analysis/visualisations/window_deformation/deformations/{os.path.splitext(filename)[0]}/{frame}_passactual.png")
            plt.close()
