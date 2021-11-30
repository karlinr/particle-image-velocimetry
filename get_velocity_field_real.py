import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import os
import time
import scipy.optimize
import msl_sad_correlation as msc
import numexpr as ne
import math


def gaussian2D(_xy, _a, _x0, _y0, _sigma_x, _sigma_y, _bg):
    (_x, _y) = _xy
    return (
    (_bg + _a * np.exp(-(((_x - _x0) ** 2 / (2 * _sigma_x ** 2)) + ((_y - _y0) ** 2 / (2 * _sigma_y ** 2)))))).ravel()


def get_particle_velocity_from_video(_filename, _iw1, _iw2, _inc):
    start = time.time()

    # Get vars
    filename = _filename
    iw1 = _iw1
    iw2 = _iw2
    inc = _inc

    # Precompute some things
    iw_d = iw2 - iw1 + 1

    # Import video and get attributes
    print(filename)
    # video = np.pad(tf.imread(_filename).astype(np.int16), ((0, 0), (iw2, iw2), (iw2, iw2)), "minimum")
    video = tf.imread(_filename).astype(np.int16)
    frames = int(video.shape[0])
    width = int(np.ceil((video.shape[1] - iw2) / inc))
    height = int(np.ceil((video.shape[2] - iw2) / inc))

    print(f"Calculating velocity field for {_filename}")
    print(f"IW1 : {iw1}; IW2 : {iw2}; Increment : {inc}")

    # Initialise arrays
    velocity_field = np.zeros((frames // 2, width, height, 4))
    mean_velocity_field = np.empty((width, height, 4))
    absolute_differences = np.empty((frames // 2, width, height, iw_d, iw_d), dtype = np.int_)
    print(absolute_differences.nbytes / 1024 / 1024)

    # Check for locations which contain particles
    image_intensity_sum = np.sum(video[::2], axis = 0)
    intensity_array = [np.sum(image_intensity_sum[j * inc: (j * inc) + iw2, k * inc: (k * inc) + iw2]) for j in
                       range(0, width) for k in range(0, height)]
    intensity_array = intensity_array - np.min(intensity_array)
    # 0.215
    intensity_array = np.array([intensity_array / np.max(intensity_array) >= 0.215]).reshape((width, height))

    # Calculate the correlation matrices for each frame pair
    print(f"- Calculating frame correlation matrices over {frames} frames")
    for i in range(0, frames, 2):
        if i % 10 == 0 and i != 0:
            print(f" - - frame {i} complete, {time.time() - start:.2f} seconds")
        b = video[i]
        a = video[i + 1]

        # Find the velocity field using the sum of absolute differences
        for j in range(0, width):
            for k in range(0, height):
                if intensity_array[j, k]:
                    # Get coordinates
                    tl_iw2_x = int(j * inc)
                    tl_iw2_y = int(k * inc)
                    br_iw2_x = int(tl_iw2_x + iw2)
                    br_iw2_y = int(tl_iw2_y + iw2)
                    tl_iw1_x = int(np.floor(tl_iw2_x + ((iw2 - iw1) / 2)))
                    tl_iw1_y = int(np.floor(tl_iw2_y + ((iw2 - iw1) / 2)))
                    br_iw1_x = int(tl_iw1_x + iw1)
                    br_iw1_y = int(tl_iw1_y + iw1)
                    x = (tl_iw1_x + br_iw1_x) / 2
                    y = (tl_iw1_y + br_iw1_y) / 2

                    # Get larger interrogation window array
                    template_to_match = b[tl_iw2_x:br_iw2_x, tl_iw2_y: br_iw2_y]
                    template = a[tl_iw1_x:br_iw1_x, tl_iw1_y:br_iw1_y]

                    # Calculate the absolute differences for the interrogation window
                    abs_diff_map = msc.sad_correlation(template.astype(int), template_to_match.astype(int))

                    # Get velocities
                    peak_position_i = np.unravel_index(abs_diff_map.argmin(), abs_diff_map.shape)
                    u = -(peak_position_i[0] - ((iw2 - iw1) / 2))
                    v = -(peak_position_i[1] - ((iw2 - iw1) / 2))

                    # Save to the arrays
                    velocity_field[i // 2, j, k, :] = [x, y, u, v]
                    absolute_differences[int(i / 2), j, k, :, :] = abs_diff_map
    print(np.max(absolute_differences))
    print(np.min(absolute_differences))
    # Calculate the mean velocity field using the time averaged array of absolute differences
    print(f" - Performing correlation averaging over {width} columns")

    mean_absolute_differences = - np.mean(absolute_differences, axis = 0)
    # fig, ax = plt.subplots(figsize = (20, 20), sharex = True)
    _x, _y = np.meshgrid(np.arange(iw_d), np.arange(iw_d))

    for j in range(0, width):
        if j % 10 == 0 and j != 0:
            print(f" - - column {j} complete, {time.time() - start:.2f} seconds")
        for k in range(0, height):
            # Get coordinates
            tl_iw2_x = int(j * inc)
            tl_iw2_y = int(k * inc)
            tl_iw1_x = int(np.floor(tl_iw2_x + ((iw2 - iw1) / 2)))
            tl_iw1_y = int(np.floor(tl_iw2_y + ((iw2 - iw1) / 2)))
            br_iw1_x = int(tl_iw1_x + iw1)
            br_iw1_y = int(tl_iw1_y + iw1)
            x = (tl_iw1_x + br_iw1_x) / 2
            y = (tl_iw1_y + br_iw1_y) / 2

            if intensity_array[j, k]:
                # Get the minima of the absolute differences to find the average velocity vector
                peak_position = np.unravel_index(mean_absolute_differences[j, k].argmax(),
                                                 mean_absolute_differences[j, k].shape)

                # Fit a gaussian
                ig_a = np.max(mean_absolute_differences[j, k]) - np.min(mean_absolute_differences[j, k])
                ig_x0 = peak_position[0]
                ig_y0 = peak_position[1]
                ig_sigma_x = 1
                ig_sigma_y = 1
                ig_bg = np.min(mean_absolute_differences[j, k])
                b_a_i = np.max(mean_absolute_differences[j, k]) - np.min(mean_absolute_differences[j, k]) - 1
                b_a_f = np.max(mean_absolute_differences[j, k]) - np.min(mean_absolute_differences[j, k]) + 1
                b_x0_i = peak_position[0] - 3
                b_x0_f = peak_position[0] + 3
                b_y0_i = peak_position[1] - 3
                b_y0_f = peak_position[1] + 3
                b_sigma_x0_i = 0
                b_sigma_x0_f = 2
                b_sigma_y0_i = 0
                b_sigma_y0_f = 2
                b_bg_i = np.min(mean_absolute_differences[j, k]) - 5
                b_bg_f = np.min(mean_absolute_differences[j, k]) + 5
                bounds = ((b_a_i, b_x0_i, b_y0_i, b_sigma_x0_i, b_sigma_y0_i, b_bg_i),
                          (b_a_f, b_x0_f, b_y0_f, b_sigma_x0_f, b_sigma_y0_f, b_bg_f))
                initial_guess = (ig_a, ig_x0, ig_y0, ig_sigma_x, ig_sigma_y, ig_bg)
                popt, pcov = scipy.optimize.curve_fit(gaussian2D, (_x, _y),
                                                      mean_absolute_differences[j, k].ravel(order = "F"),
                                                      p0 = initial_guess, bounds = bounds, maxfev = 100000)
                u_avg = -(popt[1] - ((iw2 - iw1) / 2))
                v_avg = -(popt[2] - ((iw2 - iw1) / 2))
                """fig.add_subplot(height, width, k * width + j + 1)
                plt.imshow(gaussian2D((_x, _y), *popt).reshape(mean_absolute_differences[j, k].shape, order = "F"))
                plt.imshow(mean_absolute_differences[j, k])
                plt.imshow(mean_absolute_differences[j, k] - gaussian2D((_x, _y), *popt).reshape(mean_absolute_differences[j, k].shape, order = "F"))
                plt.axis('off')"""
            else:
                u_avg = 0
                v_avg = 0

            # Save to the arrays
            mean_velocity_field[j, k, :] = [x, y, u_avg, v_avg]

    # plt.savefig(f"data/processedzebraframe1/correlation_matrix/{iw1}.png")
    # plt.show()

    absolute_differences_good = absolute_differences
    absolute_differences_bad = np.empty((frames // 2, width, height, iw_d, iw_d))
    peak_ratios = np.empty((width, height, 1))

    for i, frame in enumerate(velocity_field):
        for j in range(0, width):
            for k in range(0, height):
                mag1 = np.sqrt(frame[j, k, 2] ** 2 + frame[j, k, 3] ** 2)
                mag2 = np.sqrt(mean_velocity_field[j, k, 2] ** 2 + mean_velocity_field[j, k, 3] ** 2)
                if np.abs(mag2 - mag1) > 1:
                    absolute_differences_bad[i, j, k, :] = absolute_differences_good[i, j, k, :]
                    absolute_differences_good[i, j, k, :] = 0
    mean_absolute_differences_good = -np.mean(absolute_differences_good, axis = 0)
    mean_absolute_differences_bad = - np.mean(absolute_differences_bad, axis = 0)
    for frame in velocity_field:
        for j in range(0, width):
            for k in range(0, height):
                peak_ratios[j, k, :] = [
                    np.max(mean_absolute_differences[j, k, :]) / np.max(mean_absolute_differences_bad[j, k, :])]
                """fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                norm = Normalize(0, np.min(mean_absolute_differences[j, k, :]))
                im = cm.ScalarMappable(norm = norm)
                peak_position_i = np.unravel_index(mean_absolute_differences[j, k, :].argmax(), mean_absolute_differences[j, k, :].shape)
                u = -(peak_position_i[0] - ((iw2 - iw1) / 2))
                v = -(peak_position_i[1] - ((iw2 - iw1) / 2))
                ax1.imshow(mean_absolute_differences[j, k, :], norm = norm)
                ax1.set_title(f"Full [{u} {v}]")
                peak_position_i = np.unravel_index(mean_absolute_differences_good[j, k, :].argmax(), mean_absolute_differences_good[j, k, :].shape)
                u = -(peak_position_i[0] - ((iw2 - iw1) / 2))
                v = -(peak_position_i[1] - ((iw2 - iw1) / 2))
                ax2.imshow(mean_absolute_differences_good[j, k, :], norm = norm)
                ax2.set_title(f"Good [{u} {v}]")
                peak_position_i = np.unravel_index(mean_absolute_differences_bad[j, k, :].argmax(), mean_absolute_differences_bad[j, k, :].shape)
                u = -(peak_position_i[0] - ((iw2 - iw1) / 2))
                v = -(peak_position_i[1] - ((iw2 - iw1) / 2))
                ax3.imshow(mean_absolute_differences_bad[j, k, :], norm = norm)
                ax3.set_title(f"Bad [{u} {v}]")
                plt.colorbar(im, ax = [ax1, ax2, ax3])
                plt.show()"""

    """M = np.sqrt(mean_velocity_field[:, :, 2] ** 2 + mean_velocity_field[:, :, 3] ** 2) -3
    plt.title(f"Vector magnitude with inner interrogation window of size {iw1}px")
    plt.imshow(M, cmap = "Spectral")
    #plt.clim(0, 5)
    plt.colorbar()
    plt.savefig(f"data/analysis/{iw1}.png")
    plt.show()"""

    # np.savez_compressed(f"./data/np/{os.path.basename(os.path.splitext(filename)[0])}_{video.shape[1]}_{video.shape[2]}_{iw1}_{iw2}_{inc}_mean_field.npy", mean_velocity_field)
    # np.savez_compressed(f"./data/np/{os.path.basename(os.path.splitext(filename)[0])}_{video.shape[1]}_{video.shape[2]}_{iw1}_{iw2}_{inc}_image_intensity_sum.npy", image_intensity_sum)
    # np.savez_compressed(f"./data/np/{os.path.basename(os.path.splitext(filename)[0])}_{video.shape[1]}_{video.shape[2]}_{iw1}_{iw2}_{inc}_mean_absolute_differences.npy", mean_absolute_differences)
    # np.savez_compressed(f"./data/np/{os.path.basename(os.path.splitext(filename)[0])}_{video.shape[1]}_{video.shape[2]}_{iw1}_{iw2}_{inc}_absolute_differences.npy", absolute_differences)

    end = time.time()
    print(f"Completed in {(end - start):.2f} seconds", end = "\n\n")

    return mean_velocity_field, video.shape[1], video.shape[
        2], image_intensity_sum, mean_absolute_differences, peak_ratios


def plot_fields(_animation):
    """windows = [1, 8, 16, 32, 64, 96, 128]
    stdev = []
    for window in windows:
        field = get_particle_velocity_from_video(f"./data/animations/{_animation}", window, window+24, 24)
        mag = np.sqrt(pow(np.array(field[0][:, :, 2]), 2) + pow(np.array(field[0][:, :, 3]), 2)) - 3
        plt.imshow(mag, cmap = "Spectral")
        plt.colorbar()
        plt.show()
        stdev = np.append(stdev, np.sqrt(np.sum(np.square(mag)) / len(field[0][:, :, 3])))
    plt.plot(windows, stdev)
    plt.show()"""

    iw1 = 16
    iw2 = 48
    int = 8

    field = get_particle_velocity_from_video(f"./data/processedzebra/{_animation}", iw1, iw2, int)
    # field = get_particle_velocity_from_video(f"./data/processedzebraframe1/0_testdata.tif", iw1, iw2, int)

    x = field[0][:, :, 0]
    y = field[0][:, :, 1]
    u = field[0][:, :, 2]
    v = field[0][:, :, 3]

    plt.figure(figsize = (12, 9))
    plt.margins(iw2)
    plt.imshow(np.rot90(field[5]), cmap = "viridis", aspect = "auto",
               extent = [iw2 / 2, field[5].shape[0] * int + iw2 / 2, iw2 / 2, field[5].shape[1] * int + iw2 / 2])
    plt.colorbar()
    mag = np.sqrt(pow(np.array(field[0][:, :, 2]), 2) + pow(np.array(field[0][:, :, 3]), 2))
    # plt.quiver(x + int / 2, y + int / 2, u / mag, v / mag, mag)
    # plt.colorbar()
    plt.savefig(f"./data/analysis/err_{_animation}.png")
    plt.show()

    plt.figure(figsize = (12, 9))
    plt.title(f"Time averaged velocity field for {_animation}")
    plt.imshow(np.flip(np.flip(np.rot90(field[3])), axis = 1), cmap = "Greys", aspect = "auto")
    # plt.imshow(np.flip(np.flip(np.rot90(field[5])), axis = 1), cmap = "Greys", aspect = "auto")
    mag = np.sqrt(pow(np.array(field[0][:, :, 2]), 2) + pow(np.array(field[0][:, :, 3]), 2))
    plt.quiver(x, y, u / mag, v / mag, mag, cmap = "viridis")
    # plt.clim(0, 6)
    plt.colorbar()
    plt.xlim(0, field[1])
    plt.ylim(0, field[2])
    plt.savefig(f"./data/analysis/{_animation}.png")
    plt.show()


for animation in os.listdir("./data/processedzebra/"):
    plot_fields(animation)
