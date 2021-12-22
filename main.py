import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import time
import scipy.optimize
import msl_sad_correlation as msc
import os


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
        """

        :param name:
        :return:
        """
        return PIV.__conf[name]

    @staticmethod
    def set(name, value):
        """

        :param name:
        :param value:
        :return:
        """
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

    video = tf.imread(filename).astype(np.int16)

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

    width = int(((intensity_array.shape[0] + inc) - iw2) // inc)
    height = int(((intensity_array.shape[1] + inc) - iw2) // inc)
    intensity_array = np.array([np.sum(intensity_array[j * inc: (j * inc) + iw2, k * inc: (k * inc) + iw2]) for j in
                                range(0, width) for k in range(0, height)])
    intensity_array = intensity_array - np.min(intensity_array)
    # FIX: normalisation constant is a minimum of 1 when float values could be possible
    threshold_array = np.array([intensity_array / np.max([np.max(intensity_array), 1]) >= threshold]).reshape(
        (width, height))
    return threshold_array


def get_correlation_matrices_from_video(_threshold_array = None):
    """
    Takes the current video and threshold array and ouputs array of correlation matrices for every frame.

    :param _threshold_array:
    :return: array of absolute differences for every frame
    """
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
    width = int(((video.shape[1] + inc) - iw2) // inc)
    height = int(((video.shape[2] + inc) - iw2) // inc)

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

        # Get the absolute differences array
        for j in range(0, width):
            for k in range(0, height):
                if threshold_array[j, k]:
                    # Get coordinates
                    tl_iw2_x = int(j * inc)
                    tl_iw2_y = int(k * inc)
                    br_iw2_x = int(tl_iw2_x + iw2)
                    br_iw2_y = int(tl_iw2_y + iw2)
                    tl_iw1_x = int(tl_iw2_x + ((iw2 - iw1) // 2))
                    tl_iw1_y = int(tl_iw2_y + ((iw2 - iw1) // 2))
                    br_iw1_x = int(tl_iw1_x + iw1)
                    br_iw1_y = int(tl_iw1_y + iw1)

                    # Get interrogation windows
                    template_to_match = b[tl_iw2_x:br_iw2_x, tl_iw2_y: br_iw2_y]
                    template = a[tl_iw1_x:br_iw1_x, tl_iw1_y:br_iw1_y]

                    # Calculate the absolute differences for the interrogation window
                    absolute_differences[f // 2, j, k, :, :] = msc.sad_correlation(template.astype(int),
                                                                                   template_to_match.astype(int))

    end = time.time()
    print(f"Completed in {(end - start):.2f} seconds", end = "\n\n")
    return absolute_differences


def get_correlation_average_matrix_from_correlation_matrices(absolute_differences):
    """
    Takes the array of correlation matrices over every frame and outputs an averaged array.

    :param absolute_differences: absolute differences array for every frame
    :return: mean absolute differences array
    """
    iw1 = PIV.config("iw1")
    iw2 = PIV.config("iw2")
    width = absolute_differences.shape[1]
    height = absolute_differences.shape[2]

    mean_absolute_differences = np.empty((1, width, height, iw2 - iw1 + 1, iw2 - iw1 + 1), dtype = np.float64)
    mean_absolute_differences[0] = np.mean(absolute_differences, axis = 0)

    return mean_absolute_differences


def get_velocity_vector_from_correlation_matrix(_admatrix):
    """
    Takes a correlation matrix and returns the velocity vector.

    :param np.array _admatrix: 2d numpy array of absolute differences
    :return: u,v velocity vector tuple
    """

    # get vars
    iw1 = PIV.config("iw1")
    iw2 = PIV.config("iw2")
    inc = PIV.config("inc")

    admatrix = -_admatrix + np.max(_admatrix)
    pfmethod = PIV.config("pfmethod")

    peak_position = np.unravel_index(admatrix.argmax(), admatrix.shape)

    if pfmethod == "peak":
        u = peak_position[0] - (admatrix.shape[0] - 1) / 2
        v = peak_position[1] - (admatrix.shape[1] - 1) / 2
    elif pfmethod == "fivepointgaussian":
        if admatrix.shape[0] - 1 > peak_position[0] > 0 and admatrix.shape[1] - 1 > peak_position[1] > 0:
            xc = yc = np.log(admatrix[peak_position[0], peak_position[1]])
            xl = np.log(admatrix[peak_position[0] - 1, peak_position[1]])
            xr = np.log(admatrix[peak_position[0] + 1, peak_position[1]])
            ya = np.log(admatrix[peak_position[0], peak_position[1] - 1])
            yb = np.log(admatrix[peak_position[0], peak_position[1] + 1])
            subpixel = [(xl - xr) / (2 * (xr - 2 * xc + xl)), (ya - yb) / (2 * (yb - 2 * yc + ya))]
            u = -(peak_position[0] - (admatrix.shape[0] - 1) / 2 + subpixel[0])
            v = -(peak_position[1] - (admatrix.shape[1] - 1) / 2 + subpixel[1])
        else:
            u = -(peak_position[0] - (admatrix.shape[0] - 1) / 2)
            v = -(peak_position[1] - (admatrix.shape[1] - 1) / 2)
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
        raise ValueError("Invalid peak fitting method.")
    return u, v


def get_velocity_field_from_correlation_matrix(_correlation_matrices, _threshold):
    correlation_matrices = _correlation_matrices
    threshold = _threshold
    frames = correlation_matrices.shape[0]
    width = correlation_matrices.shape[1]
    height = correlation_matrices.shape[2]
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


def resample_correlation_matrices(absolute_differences):
    """

    :param absolute_differences:
    :return:
    """
    samples = absolute_differences.shape[0]
    indices = np.random.choice(samples, samples)
    absolute_differences_resampled = absolute_differences[indices, :, :, :]
    return absolute_differences_resampled

start = time.time()
# PIV config
PIV.set("iw1", 16)
PIV.set("iw2", 48)
PIV.set("inc", 16)
PIV.set("threshold", 0)
#PIV.set("threshold", 0.0)
PIV.set("filename", f"./data/processedzebra/0_testdata.tif")
PIV.set("pfmethod", "fivepointgaussian")
# Do PIV
intsum = get_image_intensity_sum_from_video()
thrarr = get_threshold_array_from_intensity_array(intsum)
cormat = get_correlation_matrices_from_video(thrarr)
avgmat = get_correlation_average_matrix_from_correlation_matrices(cormat)
vel_field_avg = get_velocity_field_from_correlation_matrix(avgmat, thrarr)
end = time.time()

"""for animation in os.listdir("./data/processedzebra/"):
    # PIV config
    PIV.set("filename", f"./data/processedzebra/{animation}")

    # Do PIV
    intsum = get_image_intensity_sum_from_video()
    thrarr = get_threshold_array_from_intensity_array(intsum)
    cormat = get_correlation_matrices_from_video(thrarr)
    avgmat = get_correlation_average_matrix_from_correlation_matrices(cormat)
    vel_field_avg = get_velocity_field_from_correlation_matrix(avgmat, thrarr)

    # Plot PIV
    mag = np.sqrt(pow(np.array(vel_field_avg[0][:, :, 2]), 2) + pow(np.array(vel_field_avg[0][:, :, 3]), 2))
    plt.figure(figsize = (16, 12))
    plt.imshow(np.flip(np.flip(np.rot90(intsum), axis = 1)), cmap = "Greys", aspect = "auto")
    plt.quiver(vel_field_avg[0][:, :, 0], vel_field_avg[0][:, :, 1], vel_field_avg[0][:, :, 2] / mag,
               vel_field_avg[0][:, :, 3] / mag, mag)
    #plt.clim(0, 8)
    plt.colorbar()
    plt.xlim(0, intsum.shape[0])
    plt.ylim(0, intsum.shape[1])
    plt.show()
    #plt.savefig(f"./data/deczebrafields/{animation}.png")"""

# Bootstrap
magvelsamples = []
magvel = []
velsamplesx = []
velsamplesy = []

PIV.set("filename", f"./data/processedzebra/0_testdata.tif")
intsum = get_image_intensity_sum_from_video()
thrarr = get_threshold_array_from_intensity_array(intsum)
cormat = get_correlation_matrices_from_video(thrarr)
avgmat = get_correlation_average_matrix_from_correlation_matrices(cormat)
vel_field_avg = get_velocity_field_from_correlation_matrix(avgmat, thrarr)
magvel = np.sqrt(vel_field_avg[0][:, :, 3] ** 2 + vel_field_avg[0][:, :, 2] ** 2)
start = time.time()
for i in range(10000):
    cormat_sample = resample_correlation_matrices(cormat)
    avgmat = get_correlation_average_matrix_from_correlation_matrices(cormat_sample)
    vel_field_avg = get_velocity_field_from_correlation_matrix(avgmat, thrarr)
    magvelsamples.append(np.sqrt(vel_field_avg[0][:, :, 3] ** 2 + vel_field_avg[0][:, :, 2] ** 2))
    velsamplesx.append(vel_field_avg[0][:, :, 2])
    velsamplesy.append(vel_field_avg[0][10, 10, 3])
end = time.time()
print(f"Completed in {(end - start):.2f} seconds", end = "\n\n")

magvelmean = np.mean(magvelsamples, axis = 0)
magvelmean2 = magvel
#velstd = np.std(np.sqrt(np.square(velsamplesx) + np.square(velsamplesy)), axis = 0)
"""plt.imshow(np.flip(np.flip(np.rot90(np.abs(magvelmean - magvelmean2)))), aspect = "auto", extent = [0, 320, 0, 400])
plt.colorbar()
plt.show()"""
#plt.imshow(np.flip(np.flip(np.rot90(velstd))), aspect = "auto", extent = [0, 320, 0, 400])
#plt.colorbar()
#plt.show()
"""plt.imshow(np.flip(np.flip(np.rot90(velstd / np.abs(magvelmean - magvelmean2)))), aspect = "auto", extent = [0, 320, 0, 400])
plt.clim(0, 250)
plt.colorbar()"""
#plt.show()

# Print output
print(f"magnitude velocity: {np.mean(magvelsamples)}")
print(f"magnitude x velocity: {np.mean(velsamplesx)}")
print(f"magnitude y velocity: {np.mean(velsamplesy)}")
#print(np.std(magvelsamples, ddof = 1))

# Plot bootstrap
plt.title("Bootstrapped x velocities")
plt.hist(np.array(velsamplesx).ravel(), bins = 500, align = "mid")
plt.show()
plt.title("Bootstrapped y velocities")
plt.hist(np.array(velsamplesy).ravel(), bins = 500, align = "mid")
plt.show()
plt.title("Bootstrapped speeds")
plt.hist(np.array(magvelsamples).ravel(), bins = 500, align = "mid")
plt.show()

"""for i in [16]:
    # Do PIV
    PIV.set("iw1", i)
    PIV.set("iw2", 16+i)
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
    plt.quiver(vel_field_avg[0][:, :, 0], vel_field_avg[0][:, :, 1], vel_field_avg[0][:, :, 2] / mag, vel_field_avg[0][:, :, 3] / mag, mag)
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

frames_ = [49, 49, 5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 40, 40, 30, 30, 35, 35]
ratio = [5.10 / 0.19, 4.28 / 0.20, 5.24 / 0.64, 4.55 / 0.65, 5.27 / 0.47, 5.25 / 0.46, 4.69 / 0.39, 5.23 / 0.38, 5.47 / 0.33, 4.60 / 0.34, 6.45 / 0.29, 4.09 / 0.31, 5.46 / 0.23, 5.36 / 0.21, 4.63 / 0.27, 4.96 / 0.30, 5.99 / 0.25, 4.69 / 0.22]

fit = np.polyfit(frames_, ratio, 1)
xs = np.arange(np.min(frames_), np.max(frames_), 1)
print(xs)
print(fit)
plt.plot(xs, fit[0] * xs + fit[1])

plt.scatter(frames_, ratio)
plt.ylabel("Std Dev: Std Dev Cor Avg")
plt.xlabel("Frame Pairs")
plt.show()"""
