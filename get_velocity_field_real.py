import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import os
import time


# from scipy.signal import correlate2d

# Settings
# plt.style.use('dark_background')


def gaussian():



def get_particle_velocity_from_video(_filename, _iw1, _iw2):
    start = time.time()
    print(f"Calculating velocity field for {_filename}... ")

    # Get vars
    iw1 = _iw1
    iw2 = _iw2

    # Import video and get attributes
    video = tf.imread(_filename).astype(np.int16)
    frames = int(video.shape[0])
    width = int(np.floor((video.shape[1]) / iw2))
    height = int(np.floor((video.shape[2]) / iw2))

    # Initialise arrays
    velocity_field = np.empty((int(frames / 2), width, height, 4))
    mean_velocity_field = np.empty((width, height, 4))
    absolute_differences = np.empty((int(frames / 2), width, height, iw2 - iw1, iw2 - iw1))

    for i in range(0, frames, 2):
        print(f"{i}/{frames}")
        b = video[i]
        a = video[i + 1]

        # fig, ax = plt.subplots(figsize = (20, 20), sharex = True)

        # Find the velocity field using the sum   of absolute differences
        for j in range(0, width):
            for k in range(0, height):

                # Initialise arrays
                abs_diff_map = np.zeros((iw2 - iw1, iw2 - iw1))

                # Get center of iw2
                x = int((j + 0.5) * iw2)
                y = int((k + 0.5) * iw2)

                # Get slice of image at larger interrogation window
                iw2_a = a[int(x - iw2 / 2):int(x + iw2 / 2), int(y - iw2 / 2):int(y + iw2 / 2)]
                iw2_b = b[int(x - iw2 / 2):int(x + iw2 / 2), int(y - iw2 / 2):int(y + iw2 / 2)]

                # Get slices for smaller interrogation window
                tl = int((iw2 - iw1) / 2)
                br = int((iw2 + iw1) / 2)
                iw1_a = iw2_a[tl:br, tl:br]
                iw1_b = iw2_b[tl:br, tl:br]

                # Check if there is no motion detected before attempting to find flow vectors
                """if np.array_equiv(iw1_a, iw1_b):
                    abs_diff_map[int((iw2 - iw1) / 2), int((iw2 - iw1) / 2)] = float("-inf")
                    velocity_field[i, j, k, :] = [x, y, 0, 0]
                else:"""
                # Calculate the absolute differences for the interrogation window
                for m in range(0, iw2 - iw1):
                    for n in range(0, iw2 - iw1):
                        iw1_a = iw2_a[m:m + iw1, n:n + iw1]
                        abs_diff_map[m, n] = np.sum(np.abs(iw1_a - iw1_b))

                """fig.add_subplot(width, height, k * height + j + 1)
                plt.imshow(abs_diff_map)
                plt.axis('off')"""

                # Get the minima of the absolute differences to find the velocity vector
                peak_position_i = np.unravel_index(abs_diff_map.argmin(), abs_diff_map.shape)
                u = peak_position_i[0] - ((iw2 - iw1) / 2)  # + subpixel_x
                v = peak_position_i[1] - ((iw2 - iw1) / 2)  # + subpixel_y

                # Save to the arrays
                velocity_field[int(i / 2), j, k, :] = [x, y, u, v]
                absolute_differences[int(i / 2), j, k, :, :] = abs_diff_map

        # plt.show()

    # fig, ax = plt.subplots(figsize = (20, 20), sharex = True)

    # Calculate the mean velocity field using the time averaged array of absolute differences
    mean_absolute_differences = np.mean(absolute_differences, axis=0)
    for j in range(0, width):
        for k in range(0, height):
            x = int((j + 0.5) * iw2)
            y = int((k + 0.5) * iw2)

            """fig.add_subplot(width, height, k * height + j + 1)
            plt.imshow(mean_absolute_differences[j,k])
            plt.axis('off')"""

            # Get the minima of the absolute differences to find the average velocity vector
            peak_position = np.unravel_index(mean_absolute_differences[j, k].argmin(),
                                             mean_absolute_differences[j, k].shape)
            u = peak_position[0] - ((iw2 - iw1) / 2)
            v = peak_position[1] - ((iw2 - iw1) / 2)
            u_avg = peak_position[0] - ((iw2 - iw1) / 2)
            v_avg = peak_position[1] - ((iw2 - iw1) / 2)

            # Save to the arrays
            mean_velocity_field[j, k, :] = [x, y, u_avg, v_avg]

    # plt.show()

    end = time.time()
    print(f"Completed in {(end - start):.2f} seconds")
    return velocity_field, mean_velocity_field, video.shape[1], video.shape[2], absolute_differences


def plot_fields(_animation):
    field = get_particle_velocity_from_video(f"./data/processedzebra/{_animation}", 16, 32)

    x = field[1][:, :, 0]
    y = field[1][:, :, 1]
    u = field[1][:, :, 2]
    v = field[1][:, :, 3]
    plt.title(f"Time averaged velocity field for {_animation}")
    mag = np.sqrt(pow(np.array(field[1][:, :, 2]), 2) + pow(np.array(field[1][:, :, 3]), 2))
    plt.quiver(x, y, u, v, mag, cmap="viridis")
    plt.colorbar()
    plt.xlim(0, field[2])
    plt.ylim(0, field[3])
    plt.savefig(f"./data/zebraaveraged/{_animation}.png")
    plt.show()

    x = field[0][0, :, :, 0]
    y = field[0][0, :, :, 1]
    u = field[0][0, :, :, 2]
    v = field[0][0, :, :, 3]
    plt.title(f"Velocity field for {_animation}")
    mag = np.sqrt(pow(np.array(field[0][0, :, :, 2]), 2) + pow(np.array(field[0][0, :, :, 3]), 2))
    plt.quiver(x, y, u, v, mag, cmap="viridis")
    plt.colorbar()
    plt.xlim(0, field[2])
    plt.ylim(0, field[3])
    plt.show()

    fig, ax = plt.subplots(1, 1)
    q = ax.quiver(x, y, u, v, cmap="viridis")

    def update_quiver(_f, _q):
        _q.set_UVC(field[0][_f, :, :, 2], field[0][_f, :, :, 3])
        return _q,

    anim = animate.FuncAnimation(fig, update_quiver, fargs=(q,), frames=field[0].shape[0], interval=120,
                                 blit=False)

    writervideo = animate.FFMpegWriter(fps=5)
    anim.save(f"./data/animated_fields/anim_{_animation}.mp4", writer=writervideo)
    plt.close(fig)


for animation in os.listdir("./data/processedzebra/"):
    plot_fields(animation)
