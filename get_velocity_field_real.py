import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import os
import time
import scipy.optimize


def gaussian2D(_xy, _a, _x0, _y0, _sigma_x, _sigma_y, _bg):
    (_x, _y) = _xy
    return (_bg + _a * np.exp(
        -(((_x - _x0) ** 2 / (2 * _sigma_x ** 2)) + ((_y - _y0) ** 2 / (2 * _sigma_y ** 2))))).ravel()


def get_particle_velocity_from_video(_filename, _iw1, _iw2, _inc):
    start = time.time()
    print(f"Calculating velocity field for {_filename}... ")

    # Get vars
    iw1 = _iw1
    iw2 = _iw2
    inc = _inc

    # Import video and get attributes
    video = tf.imread(_filename).astype(np.int16)
    frames = int(video.shape[0])
    width = int(np.ceil((video.shape[1] - iw2) / inc))
    height = int(np.ceil((video.shape[2] - iw2) / inc))

    # Initialise arrays
    velocity_field = np.empty((int(frames / 2), width, height, 4))
    mean_velocity_field = np.empty((width, height, 4))
    absolute_differences = np.empty((int(frames / 2), width, height, iw2 - iw1, iw2 - iw1))

    # Check for locations which contain particles
    image_intensity_sum = np.sum(video, axis = 0)
    intensity_array = [np.sum(image_intensity_sum[j * inc: (j * inc) + iw2, k * inc: (k * inc) + iw2]) for j in
                       range(0, width) for k in range(0, height)]
    intensity_array = [intensity_array / np.max(intensity_array) > 0.5]
    intensity_array = np.array(intensity_array).reshape((width, height))

    for i in range(0, frames, 2):
        print(f"{i}/{frames}")
        b = video[i]
        a = video[i + 1]
        print(f"Comparing {i} with {i + 1}")

        # Find the velocity field using the sum of absolute differences
        for j in range(0, width):
            for k in range(0, height):

                # Initialise arrays
                abs_diff_map = np.zeros((iw2 - iw1, iw2 - iw1))

                # Get center of iw2
                x = int(j * inc + 0.5 * iw2)
                y = int(k * inc + 0.5 * iw2)

                if intensity_array[j, k]:

                    # Get slice of image at larger interrogation window
                    iw2_a = a[int(x - iw2 / 2):int(x + iw2 / 2), int(y - iw2 / 2):int(y + iw2 / 2)]
                    iw2_b = b[int(x - iw2 / 2):int(x + iw2 / 2), int(y - iw2 / 2):int(y + iw2 / 2)]

                    # Get slices for smaller interrogation window
                    tl = int((iw2 - iw1) / 2)
                    br = int((iw2 + iw1) / 2)
                    iw1_b = iw2_b[tl:br, tl:br]

                    # Calculate the absolute differences for the interrogation window
                    for m in range(0, iw2 - iw1):
                        for n in range(0, iw2 - iw1):
                            iw1_a = iw2_a[m:m + iw1, n:n + iw1]
                            abs_diff_map[m, n] = np.sum(np.abs(iw1_a - iw1_b))

                    # Get the minima of the absolute differences to find the velocity vector
                    peak_position_i = np.unravel_index(abs_diff_map.argmin(), abs_diff_map.shape)
                    u = peak_position_i[0] - ((iw2 - iw1) / 2)  # + subpixel_x
                    v = peak_position_i[1] - ((iw2 - iw1) / 2)  # + subpixel_y
                else:
                    abs_diff_map = np.ones((iw2 - iw1, iw2 - iw1))
                    abs_diff_map[int((iw2 - iw1) / 2), int((iw2 - iw1) / 2)] = 0
                    u = (iw2 - iw1) / 2
                    v = (iw2 - iw1) / 2

                # Save to the arrays
                velocity_field[int(i / 2), j, k, :] = [x, y, u, v]
                absolute_differences[int(i / 2), j, k, :, :] = abs_diff_map

    # Calculate the mean velocity field using the time averaged array of absolute differences
    # Get the mean of our absolute differences array for time averaged PIV
    mean_absolute_differences = np.mean(absolute_differences, axis = 0)

    # fig, ax = plt.subplots(figsize = (20, 20), sharex = True)

    for j in range(0, width):
        for k in range(0, height):
            x = int(j * inc + iw2 / 2)
            y = int(k * inc + iw2 / 2)

            if intensity_array[j, k]:
                # Get the minima of the absolute differences to find the average velocity vector
                peak_position = np.unravel_index(mean_absolute_differences[j, k].argmin(),
                                                 mean_absolute_differences[j, k].shape)

                mean_absolute_differences[j, k] = -mean_absolute_differences[j, k]
                _x = np.arange(mean_absolute_differences[j, k].shape[0])
                _y = np.arange(mean_absolute_differences[j, k].shape[1])
                _x, _y = np.meshgrid(_x, _y)

                # Fit a gaussian
                ig_a = np.max(mean_absolute_differences[j, k]) - np.min(mean_absolute_differences[j, k])
                ig_x0 = peak_position[0]
                ig_y0 = peak_position[1]
                ig_sigma_x = 1
                ig_sigma_y = 1
                ig_bg = np.min(mean_absolute_differences[j, k])
                b_a_i = 0
                b_a_f = np.max(mean_absolute_differences[j, k]) - np.min(mean_absolute_differences[j, k])
                b_x0_i = peak_position[0] - 2
                b_x0_f = peak_position[0] + 2
                b_y0_i = peak_position[1] - 2
                b_y0_f = peak_position[1] + 2
                b_sigma_x0_i = 0
                b_sigma_x0_f = 2
                b_sigma_y0_i = 0
                b_sigma_y0_f = 2
                b_bg_i = np.min(mean_absolute_differences[j, k])
                b_bg_f = np.max(mean_absolute_differences[j, k])
                bounds = ((b_a_i, b_x0_i, b_y0_i, b_sigma_x0_i, b_sigma_y0_i, b_bg_i),
                          (b_a_f, b_x0_f, b_y0_f, b_sigma_x0_f, b_sigma_y0_f, b_bg_f))
                initial_guess = (ig_a, ig_x0, ig_y0, ig_sigma_x, ig_sigma_y, ig_bg)
                popt, pcov = scipy.optimize.curve_fit(gaussian2D, (_x, _y),
                                                      mean_absolute_differences[j, k].flatten(order = "F"),
                                                      p0 = initial_guess, bounds = bounds, maxfev = 100000)

                u_avg = popt[1] - ((iw2 - iw1) / 2)
                v_avg = popt[2] - ((iw2 - iw1) / 2)
            else:
                u_avg = 0
                v_avg = 0

            """fig.add_subplot(width, height, k * width + j + 1)
            #plt.imshow(gaussian2D((_x, _y), *popt).reshape(mean_absolute_differences[j, k].shape, order = "F"))
            plt.imshow(mean_absolute_differences[j,k])
            plt.axis('off')"""

            # Save to the arrays
            mean_velocity_field[j, k, :] = [x, y, u_avg, v_avg]

    # plt.show()

    end = time.time()
    print(f"Completed in {(end - start):.2f} seconds")
    return mean_velocity_field, video.shape[1], video.shape[2], image_intensity_sum, mean_absolute_differences


def plot_fields(_animation):
    field = get_particle_velocity_from_video(f"./data/processedzebra/{_animation}", 16, 32, 16)

    x = field[0][:, :, 0]
    y = field[0][:, :, 1]
    u = field[0][:, :, 2]
    v = field[0][:, :, 3]
    plt.title(f"Time averaged velocity field for {_animation}")
    plt.imshow(np.flip(np.flip(np.rot90(field[3])), axis = 1), cmap = "Greys", aspect = "auto")
    mag = np.sqrt(pow(np.array(field[0][:, :, 2]), 2) + pow(np.array(field[0][:, :, 3]), 2))
    plt.quiver(x, y, u / mag, v / mag, mag, cmap = "viridis")
    plt.colorbar()
    plt.xlim(0, field[1])
    plt.ylim(0, field[2])
    plt.savefig(f"./data/zebraaveraged/{_animation}.png")
    plt.show()


for animation in os.listdir("./data/processedzebra/"):
    plot_fields(animation)
