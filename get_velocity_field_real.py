import plistlib
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animate

with open("data/zebrafish/params.plist", 'rb') as fp:
    plist = plistlib.load(fp)

frames = plist["piv"]["pivGroupings"][6][0], plist["piv"]["pivGroupings"][6][1]

video_array = np.empty((2, *tf.imread(f"data/zebrafish/030298.tif").shape))

for f, fr in enumerate(frames):
    for video in os.listdir("data/zebrafish/"):
        if video.endswith(".tif"):
            current_video = tf.imread(f"data/zebrafish/{video}").astype(np.int8)
            for i, index in enumerate(fr):
                lower_index = int(os.path.splitext(video)[0])
                upper_index = int(os.path.splitext(video)[0]) + 1000
                if lower_index < index < upper_index:
                    video_array[f, i] = current_video[index - lower_index]

print(np.sum(video_array[0, 1] - video_array[1, 1]))
print(video_array.shape)


def get_particle_velocity_from_video(_video, _iw1, _iw2):
    # Get vars
    iw1 = _iw1
    iw2 = _iw2

    # Import video and get attributes
    frames = _video.shape[1]
    width = int(np.floor((_video.shape[2]) / iw2))
    height = int(np.floor((_video.shape[3]) / iw2))

    # Initialise arrays
    velocity_field = np.empty((frames, width, height, 4))
    mean_velocity_field = np.empty((width, height, 4))
    absolute_differences = np.empty((frames, width, height, iw2 - iw1, iw2 - iw1))

    for i in range(0, frames):
        print(f"{i}/{frames}")
        b = _video[0, i]
        a = _video[1, i]

        # Find the velocity field using the sum of absolute differences
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

                # Calculate the absolute differences for the interrogation window
                for m in range(0, iw2 - iw1):
                    for n in range(0, iw2 - iw1):
                        iw1_a = iw2_a[m:m + iw1, n:n + iw1]
                        abs_diff_map[m, n] = np.sum(np.abs(iw1_a - iw1_b))

                # Get the minima of the absolute differences to find the velocity vector
                peak_position_i = np.unravel_index(abs_diff_map.argmin(), abs_diff_map.shape)
                u = peak_position_i[0] - ((iw2 - iw1) / 2)
                v = peak_position_i[1] - ((iw2 - iw1) / 2)

                # Save to the arrays
                velocity_field[i, j, k, :] = [x, y, u, v]
                absolute_differences[i, j, k, :, :] = abs_diff_map

    # Calculate the mean velocity field using the time averaged array of absolute differences
    mean_absolute_differences = np.mean(absolute_differences, axis = 0)
    for j in range(0, width):
        for k in range(0, height):
            x = int((j + 0.5) * iw2)
            y = int((k + 0.5) * iw2)

            # Get the minima of the absolute differences to find the average velocity vector
            peak_position = np.unravel_index(mean_absolute_differences[j, k].argmin(),
                                             mean_absolute_differences[j, k].shape)
            u = peak_position[0] - ((iw2 - iw1) / 2)
            v = peak_position[1] - ((iw2 - iw1) / 2)
            u_avg = peak_position[0] - ((iw2 - iw1) / 2)
            v_avg = peak_position[1] - ((iw2 - iw1) / 2)

            # Save to the arrays
            mean_velocity_field[j, k, :] = [x, y, u_avg, v_avg]

    return velocity_field, mean_velocity_field, _video.shape[2], _video.shape[3], absolute_differences


field = get_particle_velocity_from_video(video_array[:,:,:,:], 22, 32)

x = field[0][0, :, :, 0]
y = field[0][0, :, :, 1]
u = field[0][0, :, :, 2]
v = field[0][0, :, :, 3]
plt.title("velocity field")
mag = np.sqrt(pow(np.array(field[1][:, :, 2]), 2) + pow(np.array(field[1][:, :, 3]), 2))
plt.quiver(x, y, u, v, mag, cmap = "viridis")
plt.colorbar()
plt.xlim(0, field[2])
plt.ylim(0, field[3])
plt.show()

x = field[1][:, :, 0]
y = field[1][:, :, 1]
u = field[1][:, :, 2]
v = field[1][:, :, 3]
plt.title("Time averaged velocity field")
mag = np.sqrt(pow(np.array(field[1][:, :, 2]), 2) + pow(np.array(field[1][:, :, 3]), 2))
plt.quiver(x, y, u, v, mag, cmap = "viridis")
plt.colorbar()
plt.xlim(0, field[2])
plt.ylim(0, field[3])
plt.show()