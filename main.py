import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import os
import time
import scipy.optimize
import msl_sad_correlation as msc


class PIV:
    # https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py
    __conf = {
        "filename": "",
        "iw1": 16,
        "iw2": 48,
        "inc": 16,
        "threshold": 0,
        "pfmethod": "minima",
    }
    __setters = ["filename", "iw1", "iw2", "inc", "threshold", "pfmethod"]

    @staticmethod
    def config(name):
        return PIV.__conf[name]

    @staticmethod
    def set(name, value):
        if name in PIV.__setters:
            PIV.__conf[name] = value
            print(f"{name} set to {PIV.__conf[name]}")
        else:
            raise NameError("Name not accepted in set() method")


def gaussian2D(_xy, _a, _x0, _y0, _sigma_x, _sigma_y, _bg):
    (_x, _y) = _xy
    return (
        (_bg + _a * np.exp(
            -(((_x - _x0) ** 2 / (2 * _sigma_x ** 2)) + ((_y - _y0) ** 2 / (2 * _sigma_y ** 2)))))).ravel()


def get_image_intensity_sum_from_video():
    # Get vars
    filename = PIV.config("filename")
    iw2 = PIV.config("iw2")
    inc = PIV.config("inc")

    video = tf.imread(filename).astype(np.int16)
    width = int(np.ceil((video.shape[1] - iw2) / inc))
    height = int(np.ceil((video.shape[2] - iw2) / inc))

    intensity_array = np.sum(video[::2], axis = 0)

    return intensity_array


def get_threshold_array_from_intensity_array(intensity_array):
    """
    :param intensity_array:
    :return threshold_array:
    """
    # Get vars
    threshold = PIV.config("threshold")
    iw2 = PIV.config("iw2")
    inc = PIV.config("inc")

    width = int(np.ceil((intensity_array.shape[0] - iw2) / inc))
    height = int(np.ceil((intensity_array.shape[1] - iw2) / inc))

    intensity_array = [np.sum(intensity_array[j * inc: (j * inc) + iw2, k * inc: (k * inc) + iw2]) for j in
                       range(0, width) for k in range(0, height)]
    intensity_array = intensity_array - np.min(intensity_array)
    threshold_array = np.array([intensity_array / np.max(intensity_array) >= threshold]).reshape((width, height))

    return threshold_array


def get_correlation_matrices_from_video(_threshold_array = None):
    # Get vars
    filename = PIV.config("filename")
    iw1 = PIV.config("iw1")
    iw2 = PIV.config("iw2")
    inc = PIV.config("inc")

    # Time and out
    start = time.time()
    print(f"Calculating correlation matrices {filename}")

    # Import video and get attributes
    video = tf.imread(filename).astype(np.int16)
    frames = int(video.shape[0])
    width = int(np.ceil((video.shape[1] - iw2) / inc))
    height = int(np.ceil((video.shape[2] - iw2) / inc))

    # Get threshold array
    if _threshold_array is None:
        threshold_array = np.ones(video[0].shape)
    else:
        threshold_array = _threshold_array

    # Initialise arrays
    absolute_differences = np.empty((frames // 2, width, height, iw2 - iw1 + 1, iw2 - iw1 + 1), dtype = np.int_)

    # Calculate the correlation matrices for each frame pair
    print(f"- Calculating frame correlation matrices over {frames} frames")
    for f in range(0, frames, 2):
        if f % 10 == 0 and f != 0:
            print(f" - - frame {f} complete")
        b = video[f]
        a = video[f + 1]

        # Find the velocity field using the sum of absolute differences
        for j in range(0, width):
            for k in range(0, height):
                if threshold_array[j, k]:
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
                    absolute_differences[f // 2, j, k, :, :] = msc.sad_correlation(template.astype(int),
                                                                                   template_to_match.astype(int))

    end = time.time()
    print(f"Completed in {(end - start):.2f} seconds", end = "\n\n")
    return absolute_differences


def get_correlation_average_matrix_from_correlation_matrices(absolute_differences):
    iw1 = PIV.config("iw1")
    iw2 = PIV.config("iw2")
    width = absolute_differences.shape[1]
    height = absolute_differences.shape[2]

    mean_absolute_differences = np.empty((1, width, height, iw2 - iw1 + 1, iw2 - iw1 + 1), dtype = np.int_)
    mean_absolute_differences[0] = np.mean(absolute_differences, axis = 0)

    return mean_absolute_differences


def get_velocity_vector_from_correlation_matrix(_admatrix):
    # Takes a correlation matrix and finds velocity vector
    # _admatrix - absolute difference matrix
    # pfmethod - Peak finding method
    #   "minima"            -   Uses the minima of the correlation matrix
    #   "fivepointgaussian" -   Uses five-point Gaussian interpolation
    #   "gaussian"          -   Uses a Gaussian fit

    admatrix = _admatrix
    pfmethod = PIV.config("pfmethod")

    peak_position = np.unravel_index(admatrix.argmin(), admatrix.shape)

    if pfmethod == "peak":
        u = peak_position[0]- admatrix.shape[0] / 2
        v = peak_position[1] - admatrix.shape[0] / 2
    elif pfmethod == "fivepointgaussian":
        if admatrix.shape[0] - 1 > peak_position[0] > 0 and admatrix.shape[1] - 1 > peak_position[1] > 0:
            xc = yc = np.log(admatrix[peak_position[0], peak_position[1]])
            xl = np.log(admatrix[peak_position[0] - 1, peak_position[1]])
            xr = np.log(admatrix[peak_position[0] + 1, peak_position[1]])
            ya = np.log(admatrix[peak_position[0], peak_position[1] - 1])
            yb = np.log(admatrix[peak_position[0], peak_position[1] + 1])
            subpixel = [(xl - xr) / (2 * (xr - 2 * xc + xl)), (ya - yb) / (2 * (yb - 2 * yc + ya))]
            u = peak_position[0] - (admatrix.shape[0] - 1) / 2 + subpixel[0]
            v = peak_position[1] - (admatrix.shape[1] - 1) / 2 + subpixel[1]
        else:
            u = peak_position[0] - (admatrix.shape[0] - 1) / 2
            v = peak_position[1] - (admatrix.shape[0] - 1) / 2
    elif pfmethod == "gaussian":
        # Fit a gaussian
        _x, _y = np.meshgrid(np.arange(admatrix.shape[0]), np.arange(admatrix.shape[1]))

        # Initial values
        ig_a = -np.max(admatrix)
        ig_x0 = peak_position[0]
        ig_y0 = peak_position[1]
        ig_sigma_x = 1
        ig_sigma_y = 1
        ig_bg = np.max(admatrix)
        initial_guess = (ig_a, ig_x0, ig_y0, ig_sigma_x, ig_sigma_y, ig_bg)

        # Boundaries
        b_a_i = -np.max(admatrix) - 5
        b_a_f = 0
        b_x0_i = peak_position[0] - 1
        b_x0_f = peak_position[0] + 1
        b_y0_i = peak_position[1] - 1
        b_y0_f = peak_position[1] + 1
        b_sigma_x0_i = 0
        b_sigma_x0_f = 2
        b_sigma_y0_i = 0
        b_sigma_y0_f = 2
        b_bg_i = np.max(admatrix) - 5
        b_bg_f = np.max(admatrix) + 5
        bounds = ((b_a_i, b_x0_i, b_y0_i, b_sigma_x0_i, b_sigma_y0_i, b_bg_i),
                  (b_a_f, b_x0_f, b_y0_f, b_sigma_x0_f, b_sigma_y0_f, b_bg_f))
        # Do the fit
        popt, pcov = scipy.optimize.curve_fit(gaussian2D, (_x, _y),
                                              admatrix.ravel(order = "F"),
                                              p0 = initial_guess, bounds = bounds, maxfev = 100000)
        u = popt[1] - (admatrix.shape[0] - 1) / 2
        v = popt[2] - (admatrix.shape[1] - 1) / 2
    else:
        print("Invalid method.")
        u = 0
        v = 0
    return u, v


def get_velocity_field_from_correlation_matrix(_correlation_matrices, _threshold):
    correlation_matrices = _correlation_matrices
    threshold = _threshold
    frames = correlation_matrices.shape[0]
    width = correlation_matrices.shape[1]
    height = correlation_matrices.shape[2]
    iw1 = PIV.config("iw1")
    iw2 = PIV.config("iw2")
    inc = PIV.config("inc")
    velocity_field = np.zeros((frames, width, height, 4))

    for f in range(0, frames):
        if f % 10 == 0 and f != 0:
            print(f" - - frame {f} complete")
        for j in range(0, width):
            for k in range(0, height):
                if threshold[j, k]:
                    x = j * inc + iw2 / 2
                    y = k * inc + iw2 / 2
                    velocity_field[f, j, k, :] = [x, y, *get_velocity_vector_from_correlation_matrix(
                        correlation_matrices[f, j, k])]

    return velocity_field


# PIV config
PIV.set("filename", "./data/animations/animation_constant_300.tif")
PIV.set("iw1", 16)
PIV.set("iw2", 48)
PIV.set("inc", 6)
PIV.set("threshold", 0.215)
PIV.set("pfmethod", "fivepointgaussian")

# Do PIV
intsum = get_image_intensity_sum_from_video()
thrarr = get_threshold_array_from_intensity_array(intsum)
cormat = get_correlation_matrices_from_video(thrarr)
avgmat = get_correlation_average_matrix_from_correlation_matrices(cormat)
vel_field = get_velocity_field_from_correlation_matrix(cormat, thrarr)
vel_field_avg = get_velocity_field_from_correlation_matrix(avgmat, thrarr)

# Plot
mag = np.sqrt(pow(np.array(vel_field_avg[0][:, :, 2]), 2) + pow(np.array(vel_field_avg[0][:, :, 3]), 2))
plt.figure(figsize = (16, 12))
plt.imshow(np.flip(np.flip(np.rot90(intsum), axis = 1)), cmap = "Greys", aspect = "auto")
plt.quiver(vel_field_avg[0][:, :, 0], vel_field_avg[0][:, :, 1], vel_field_avg[0][:, :, 2] / mag,vel_field_avg[0][:, :, 3] / mag, mag)
plt.colorbar()
plt.xlim(0, intsum.shape[0])
plt.ylim(0, intsum.shape[1])
plt.show()

means = []
stds = []

for i in range(vel_field.shape[0]):
    means.append((np.nanmean(vel_field[i][thrarr, 2]), np.nanmean(vel_field[i][thrarr, 3])))
    stds.append((np.nanstd(vel_field[i][thrarr, 2]), np.nanstd(vel_field[i][thrarr, 2])))

print("Non-correlation Averaged")
print(f"x Velocity:{np.mean(means[0]):.2f} +/- {np.mean(stds[0]):.2f}")
print(f"y Velocity:{np.mean(means[1]):.2f} +/- {np.mean(stds[1]):.2f}")

print("Correlation Averaged")
print(f"x Velocity:{np.mean(vel_field_avg[0][thrarr, 2]):.2f} +/- {np.std(vel_field_avg[0][thrarr, 2]):.2f}")
print(f"y Velocity:{np.mean(vel_field_avg[0][thrarr, 3]):.2f} +/- {np.std(vel_field_avg[0][thrarr, 3]):.2f}")
plt.hist(vel_field_avg[0][thrarr, 2].ravel(), bins = 70)
plt.show()
plt.hist(vel_field_avg[0][thrarr, 3].ravel(), bins = 70)
plt.show()

frames = [49, 49, 5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 40, 40, 30, 30, 35, 35, 150, 150]
ratio = [5.10 / 0.19, 4.28 / 0.20, 5.24 / 0.64, 4.55 / 0.65, 5.27 / 0.47, 5.25 / 0.46, 4.69 / 0.39, 5.23 / 0.38, 5.47 / 0.33, 4.60 / 0.34, 6.45 / 0.29, 4.09 / 0.31, 5.46 / 0.23, 5.36 / 0.21, 4.63 / 0.27, 4.96 / 0.30, 5.99 / 0.25, 4.69 / 0.22, 0, 0]

fit = np.polyfit(frames, ratio, 1)
xs = np.arange(np.min(frames), np.max(frames), 1)
print(xs)
print(fit)
plt.plot(xs, fit[0] * xs + fit[1])

plt.scatter(frames, ratio)
plt.ylabel("Std Dev: Std Dev Cor Avg")
plt.xlabel("Frame Pairs")
plt.show()