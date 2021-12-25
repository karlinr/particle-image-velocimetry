import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import time
import scipy.optimize
# import msl_sad_correlation as msc
import os


class PIV:
    def __init__(self, _filename, _iw1, _iw2,- _inc, _threshold, _pfmethod):
        """

        :param _filename:
        :param _iw1:
        :param _iw2:
        :param _inc:
        :param _threshold:
        :param _pfmethod:
        """
        # Get config variables
        self.filename = _filename
        self.iw1 = _iw1
        self.iw2 = _iw2
        self.inc = _inc
        self.threshold = _threshold
        self.pfmethod = _pfmethod
        self.video = tf.imread(self.filename).astype(np.int16)
        self.frames = int(self.video.shape[0])
        self.width = int(((self.video.shape[1] + self.inc) - self.iw2) // self.inc)
        self.height = int(((self.video.shape[2] + self.inc) - self.iw2) // self.inc)

        # Set init variables
        self.intensity_array = None
        self.threshold_array = None
        self.correlation_matrices = None
        self.correlation_averaged = None
        self.velocity_field = None
        self.correlation_averaged_velocity_field = None
        self.resampled_correlation_averaged_velocity_field = None

        # Run correlation averaged PIV
        self.run_PIV()

    def run_PIV(self):
        self.get_image_intensity_sum()
        self.get_threshold_array()
        self.get_correlation_matrices()
        self.get_correlation_averaged()
        self.get_correlation_averaged_velocity_field()

    def get_image_intensity_sum(self):
        print("Getting image intensity sum...", end = " ")
        self.intensity_array = np.sum(self.video[::2], axis = 0)
        print("complete")

    def get_threshold_array(self):
        print("Getting threshold array...", end = " ")
        intensity_array = np.array(
            [np.sum(self.intensity_array[j * self.inc: (j * self.inc) + self.iw2, k * self.inc: (k * self.inc) + self.iw2]) for j in range(0, self.width) for k
             in range(0, self.height)])
        intensity_array = intensity_array - np.min(intensity_array)
        # FIX: normalisation constant is a minimum of 1 when float values could be possible
        self.threshold_array = np.array([intensity_array / np.max([np.max(intensity_array), 1]) >= self.threshold]).reshape((self.width, self.height))
        print("complete")

    def get_correlation_matrices(self):
        print("Getting correlation matrices...", end = " ")
        # Initialise arrays
        self.correlation_matrices = np.empty((self.frames // 2, self.width, self.height, self.iw2 - self.iw1 + 1, self.iw2 - self.iw1 + 1), dtype = np.int_)
        # Calculate the correlation matrices for each frame pair
        for f in range(0, self.frames, 2):
            b = self.video[f]
            a = self.video[f + 1]

            # Get the absolute differences array
            for j in range(0, self.width):
                for k in range(0, self.height):
                    if self.threshold_array[j, k]:
                        # Get coordinates
                        tl_iw2_x = int(j * self.inc)
                        tl_iw2_y = int(k * self.inc)
                        br_iw2_x = int(tl_iw2_x + self.iw2)
                        br_iw2_y = int(tl_iw2_y + self.iw2)
                        tl_iw1_x = int(tl_iw2_x + ((self.iw2 - self.iw1) // 2))
                        tl_iw1_y = int(tl_iw2_y + ((self.iw2 - self.iw1) // 2))
                        br_iw1_x = int(tl_iw1_x + self.iw1)
                        br_iw1_y = int(tl_iw1_y + self.iw1)

                        # Get interrogation windows
                        template_to_match = b[tl_iw2_x:br_iw2_x, tl_iw2_y: br_iw2_y]
                        template = a[tl_iw1_x:br_iw1_x, tl_iw1_y:br_iw1_y]

                        # Calculate the absolute differences for the interrogation window
                        # TODO: Temp fix until msc.sad_correlation is working - very slow - make faster using array views
                        # self.correlation_matrices[f // 2, j, k, :, :] = msc.sad_correlation(template.astype(int), template_to_match.astype(int))
                        abs_diff_map = np.array([np.sum(np.abs(template_to_match[m:m + self.iw1, n:n + self.iw1] - template)) for m in range(0, self.iw2 - self.iw1 + 1) for n in range(0, self.iw2 - self.iw1 + 1)]).reshape([self.iw2 - self.iw1 + 1, self.iw2 - self.iw1 + 1])
                        self.correlation_matrices[f // 2, j, k, :, :] = abs_diff_map
        print("complete")

    def get_correlation_averaged(self):
        print("Getting correlation averaged matrix...", end = " ")
        self.correlation_averaged = np.empty((1, self.width, self.height, self.iw2 - self.iw1 + 1, self.iw2 - self.iw1 + 1), dtype = np.float64)
        self.correlation_averaged[0] = np.mean(self.correlation_matrices, axis = 0)
        print("complete")

    def get_velocity_vector_from_correlation_matrix(self, _correlation_matrix):
        correlation_matrix = -_correlation_matrix + np.max(_correlation_matrix)
        peak_position = np.unravel_index(correlation_matrix.argmax(), correlation_matrix.shape)

        if self.pfmethod == "peak":
            u = peak_position[0] - (correlation_matrix.shape[0] - 1) / 2
            v = peak_position[1] - (correlation_matrix.shape[1] - 1) / 2
        elif self.pfmethod == "fivepointgaussian":
            # Todo: seperate if statement so gaussian interpolation is calculated for each separately.
            if correlation_matrix.shape[0] - 1 > peak_position[0] > 0 and correlation_matrix.shape[1] - 1 > peak_position[1] > 0:
                xc = yc = np.log(correlation_matrix[peak_position[0], peak_position[1]])
                xl = np.log(correlation_matrix[peak_position[0] - 1, peak_position[1]])
                xr = np.log(correlation_matrix[peak_position[0] + 1, peak_position[1]])
                ya = np.log(correlation_matrix[peak_position[0], peak_position[1] - 1])
                yb = np.log(correlation_matrix[peak_position[0], peak_position[1] + 1])
                subpixel = [(xl - xr) / (2 * (xr - 2 * xc + xl)), (ya - yb) / (2 * (yb - 2 * yc + ya))]
                u = -(peak_position[0] - (correlation_matrix.shape[0] - 1) / 2 + subpixel[0])
                v = -(peak_position[1] - (correlation_matrix.shape[1] - 1) / 2 + subpixel[1])
            else:
                u = -(peak_position[0] - (correlation_matrix.shape[0] - 1) / 2)
                v = -(peak_position[1] - (correlation_matrix.shape[1] - 1) / 2)
        elif self.pfmethod == "gaussian":
            # Todo: write gaussian fitting routine
            def gaussian2D(_xy, _a, _x0, _y0, _sigma_x, _sigma_y, _bg):
                (_x, _y) = _xy
                return (_bg + _a * np.exp(-(((_x - _x0) ** 2 / (2 * _sigma_x ** 2)) + ((_y - _y0) ** 2 / (2 * _sigma_y ** 2))))).ravel()

            return 0
        else:
            raise ValueError("Invalid peak fitting method.")
        return u, v

    def get_velocity_field_from_correlation_matrices(self, correlation_matrix):
        velocity_field = np.zeros((self.frames, self.width, self.height, 4))

        for f in range(0, correlation_matrix.shape[0]):
            for j in range(0, self.width):
                for k in range(0, self.height):
                    if self.threshold_array[j, k]:
                        x = j * self.inc + self.iw2 / 2
                        y = k * self.inc + self.iw2 / 2
                        velocity_field[f, j, k, :] = [x, y, *self.get_velocity_vector_from_correlation_matrix(correlation_matrix[f, j, k])]
        return velocity_field

    def get_velocity_field(self):
        print("Getting velocity field for all frames...", end = " ")
        self.velocity_field = self.get_velocity_field_from_correlation_matrices(self.correlation_matrices)
        print("complete")

    def get_correlation_averaged_velocity_field(self):
        print("Getting correlation averaged velocity field", end = " ")
        self.correlation_averaged_velocity_field = self.get_velocity_field_from_correlation_matrices(self.correlation_averaged)
        print("complete")

    def get_resampled_correlation_averaged_velocity_field(self):
        # Todo: Vectorise this
        # Todo: -- random.choice extra dimension for number of samples
        # Todo: -- return extra dimension in resampled averaged field like with get_velocity_field
        indices = np.random.choice(self.frames // 2, self.frames // 2)
        resampled_correlation_matrices = self.correlation_matrices[indices, :, :, :]
        resampled_correlation_matrices_averaged = np.empty((1, self.width, self.height, self.iw2 - self.iw1 + 1, self.iw2 - self.iw1 + 1), dtype = np.float64)
        resampled_correlation_matrices_averaged[0] = np.mean(resampled_correlation_matrices, axis = 0)
        self.resampled_correlation_averaged_velocity_field = self.get_velocity_field_from_correlation_matrices(resampled_correlation_matrices_averaged)


# Do PIV
start = time.time()
#pivtest = PIV("./data/processedzebra/29_testdata.tif", 16, 48, 64, 0.215, "fivepointgaussian")
pivtest = PIV("./data/animations/animation_constant_with_gradient.tif", 36, 54, 1, 0, "fivepointgaussian")
end = time.time()
print(f"Completed in {(end - start):.2f} seconds", end = "\n\n")

# Plot velocity field
mag = np.sqrt(np.square(np.array(pivtest.correlation_averaged_velocity_field[0][:, :, 2])) + np.square(np.array(pivtest.correlation_averaged_velocity_field[0][:, :, 3])))
plt.figure(figsize = (16, 12))
plt.imshow(np.flip(np.flip(np.rot90(pivtest.intensity_array), axis = 1)), cmap = "Greys", aspect = "auto")
plt.quiver(pivtest.correlation_averaged_velocity_field[0][:, :, 0], pivtest.correlation_averaged_velocity_field[0][:, :, 1],
           pivtest.correlation_averaged_velocity_field[0][:, :, 2] / mag,
           pivtest.correlation_averaged_velocity_field[0][:, :, 3] / mag, mag)
plt.clim(0, 8)
plt.colorbar()
plt.xlim(0, pivtest.intensity_array.shape[0])
plt.ylim(0, pivtest.intensity_array.shape[1])
plt.show()

# Get bootstrap
samples = 10000
vels = []
vels = np.empty((samples, pivtest.width, pivtest.height), dtype = np.float64)

start = time.time()
for i in range(samples):
    pivtest.get_resampled_correlation_averaged_velocity_field()
    vels[i, :, :] = pivtest.resampled_correlation_averaged_velocity_field[0][:, :, 2]
end = time.time()
print(f"Completed in {(end - start):.2f} seconds", end = "\n\n")

# Plot histograms
start = time.time()
fig, ax = plt.subplots(figsize = (20, 20), sharex = True)
plt.axis('off')

for j in range(0, pivtest.width):
    for k in range(0, pivtest.height):
        if pivtest.threshold_array[j, k]:
            fig.add_subplot(pivtest.height, pivtest.width, k * pivtest.width + j + 1)
            plt.hist(vels[:, j, k].ravel(), bins = 200)
            plt.axvline(pivtest.correlation_averaged_velocity_field[0][j, k, 2], c = "crimson")
plt.show()
print(f"Completed in {(end - start):.2f} seconds", end = "\n\n")