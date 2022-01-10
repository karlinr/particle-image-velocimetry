import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import scipy.optimize
import msl_sad_correlation as msc
import os


class PIV:
    def __init__(self, _filename, _iw, _sa, _inc, _threshold, _pfmethod, _pad = False):
        """

        :param _filename:
        :param _iw:
        :param _sa:
        :param _inc:
        :param _threshold:
        :param _pfmethod:
        :param _pad:
        """
        # Get config variables
        self.filename = _filename
        self.iw = _iw
        self.sa = _sa
        self.inc = _inc
        self.threshold = _threshold
        self.pfmethod = _pfmethod
        self.video = tf.imread(self.filename).astype(np.int16)
        if _pad:
            self.video = np.pad(self.video, [(0, 0), (int(self.sa + 0.5 * self.iw), int(self.sa + 0.5 * self.iw)), (int(self.sa + 0.5 * self.iw), int(self.sa + 0.5 * self.iw))], mode = "minimum")
        self.frames = int(self.video.shape[0])
        self.width = int(((self.video.shape[1] + self.inc) - (2 * self.sa + self.iw)) // self.inc)
        self.height = int(((self.video.shape[2] + self.inc) - (2 * self.sa + self.iw)) // self.inc)

        # Set init variables
        self.intensity_array = None
        self.threshold_array = None
        self.correlation_matrices = None
        self.correlation_averaged = None
        self.velocity_field = None
        self.correlation_averaged_velocity_field = None
        self.resampled_correlation_averaged_velocity_field = None

        # Run correlation averaged PIV
        print(f"Running PIV for {self.filename} with IW:{self.iw}, SA:{self.sa}, inc:{self.inc}, threshold:{self.threshold}, pfmethod: {self.pfmethod}")
        self.run_PIV()
        print("Complete", end = "\n\n")

    def run_PIV(self):
        self.get_image_intensity_sum()
        self.get_threshold_array()
        self.get_correlation_matrices()
        self.get_correlation_averaged()
        self.get_correlation_averaged_velocity_field()

    def get_image_intensity_sum(self):
        print("--Getting image intensity sum...", end = " ")
        self.intensity_array = np.sum(self.video[::2], axis = 0)
        print("complete")

    def get_threshold_array(self):
        print("--Getting threshold array...", end = " ")
        intensity_array = np.array(
            [np.sum(self.intensity_array[j * self.inc: (j * self.inc) + (2 * self.sa + self.iw), k * self.inc: (k * self.inc) + (2 * self.sa + self.iw)]) for j in range(0, self.width) for k
             in range(0, self.height)])
        intensity_array = intensity_array - np.min(intensity_array)
        # FIX: normalisation constant is a minimum of 1 when float values could be possible
        self.threshold_array = np.array([intensity_array / np.max([np.max(intensity_array), 1]) >= self.threshold]).reshape((self.width, self.height))
        print("complete")

    def get_correlation_matrices(self):
        print("--Getting correlation matrices...", end = " ")
        # Initialise arrays
        self.correlation_matrices = np.empty((self.frames // 2, self.width, self.height, 2 * self.sa + 1, 2 * self.sa + 1), dtype = np.int_)

        # Calculate the correlation matrices for each frame pair
        for f in range(0, self.frames, 2):
            # Get frames
            b = self.video[f]
            a = self.video[f + 1]

            # Get the absolute differences array
            for j in range(0, self.width):
                for k in range(0, self.height):
                    if self.threshold_array[j, k]:
                        # Get coordinates
                        tl_iw2_x = int(j * self.inc)
                        tl_iw2_y = int(k * self.inc)
                        tl_iw1_x = int(tl_iw2_x + self.sa)
                        tl_iw1_y = int(tl_iw2_y + self.sa)
                        br_iw1_x = int(tl_iw1_x + self.iw)
                        br_iw1_y = int(tl_iw1_y + self.iw)
                        br_iw2_x = int(br_iw1_x + self.sa)
                        br_iw2_y = int(br_iw1_y + self.sa)

                        # Get interrogation windows
                        template_to_match = b[tl_iw2_x:br_iw2_x, tl_iw2_y: br_iw2_y]
                        template = a[tl_iw1_x:br_iw1_x, tl_iw1_y:br_iw1_y]

                        # Calculate the absolute differences for the interrogation window
                        self.correlation_matrices[f // 2, j, k, :, :] = msc.sad_correlation(template.astype(int), template_to_match.astype(int))
                        # abs_diff_map = np.array([np.sum(np.abs(template_to_match[m:m + self.iw, n:n + self.iw] - template)) for m in range(0, 2 * self.sa + 1) for n in range(0, 2 * self.sa + 1)]).reshape([2 * self.sa + 1, 2 * self.sa + 1])
                        # self.correlation_matrices[f // 2, j, k, :, :] = abs_diff_map
        print("complete")

    def get_correlation_averaged(self):
        print("--Getting correlation averaged matrix...", end = " ")
        self.correlation_averaged = np.empty((1, *self.correlation_matrices.shape[1:5]), dtype = np.float64)
        self.correlation_averaged[0] = np.mean(self.correlation_matrices, axis = 0)
        print("complete")

    def get_velocity_vector_from_correlation_matrix(self, _correlation_matrix):
        correlation_matrix = -_correlation_matrix + np.max(_correlation_matrix)
        peak_position = np.unravel_index(correlation_matrix.argmax(), correlation_matrix.shape)

        # methods from : https://www.aa.washington.edu/sites/aa/files/faculty/dabiri/pubs/piV.Review.Paper.final.pdf
        if self.pfmethod == "peak":
            u = peak_position[0] - (correlation_matrix.shape[0] - 1) / 2
            v = peak_position[1] - (correlation_matrix.shape[1] - 1) / 2
        elif self.pfmethod == "fivepointgaussian":
            # TODO: seperate if statement so gaussian interpolation is calculated for each separately.
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
        elif self.pfmethod == "parabolic":
            xc = yc = correlation_matrix[peak_position[0], peak_position[1]]
            xl = correlation_matrix[peak_position[0] - 1, peak_position[1]]
            xr = correlation_matrix[peak_position[0] + 1, peak_position[1]]
            ya = correlation_matrix[peak_position[0], peak_position[1] - 1]
            yb = correlation_matrix[peak_position[0], peak_position[1] + 1]
            subpixel = [(xl - xr) / (2 * (xl + xr - 2 * xc)), (ya - yb) / (2 * (ya + yb - 2 * yc))]
            u = -(peak_position[0] - (correlation_matrix.shape[0] - 1) / 2 + subpixel[0])
            v = -(peak_position[1] - (correlation_matrix.shape[1] - 1) / 2 + subpixel[1])
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
                        x = j * self.inc + self.sa + 0.5 * self.iw
                        y = k * self.inc + self.sa + 0.5 * self.iw
                        velocity_field[f, j, k, :] = [x, y, *self.get_velocity_vector_from_correlation_matrix(correlation_matrix[f, j, k])]
        return velocity_field

    def get_velocity_field(self):
        print("--Getting velocity field for all frames...", end = " ")
        self.velocity_field = self.get_velocity_field_from_correlation_matrices(self.correlation_matrices)
        print("complete")

    def get_correlation_averaged_velocity_field(self):
        print("--Getting correlation averaged velocity field", end = " ")
        self.correlation_averaged_velocity_field = self.get_velocity_field_from_correlation_matrices(self.correlation_averaged)
        print("complete")

    def get_resampled_correlation_averaged_velocity_field(self):
        # Todo: Vectorise this
        # Todo: -- random.choice extra dimension for number of samples
        # Todo: -- return extra dimension in resampled averaged field like with get_velocity_field
        indices = np.random.choice(self.frames // 2, self.frames // 2)
        resampled_correlation_matrices = self.correlation_matrices[indices, :, :, :]
        resampled_correlation_matrices_averaged = np.empty((1, self.width, self.height, 2 * self.sa + 1, 2 * self.sa + 1), dtype = np.float64)
        resampled_correlation_matrices_averaged[0] = np.mean(resampled_correlation_matrices, axis = 0)
        self.resampled_correlation_averaged_velocity_field = self.get_velocity_field_from_correlation_matrices(resampled_correlation_matrices_averaged)


# Get distribution
"""v_x = np.empty(1000, dtype = np.float64)
v_y = np.empty(1000, dtype = np.float64)

for i, filename in enumerate(os.listdir("./data/animations/gradient")):
    # Plot velocity field
    pivtest = PIV(f"./data/animations/gradient/{filename}", 24, 15, 1, 0, "fivepointgaussian", False)
    v_x[i] = pivtest.correlation_averaged_velocity_field[0][:, :, 2]
    v_y[i] = pivtest.correlation_averaged_velocity_field[0][:, :, 3]

print(np.mean(v_x))
print(np.mean(v_y))

plt.figure(figsize = (16, 16))
plt.hist2d(v_x, v_y, 100)
plt.axvline(5.5)
plt.axhline(0)
plt.show()"""


# Bootstrap Test
for i, filename in enumerate(os.listdir("./data/animations/constant_10_100_1000")):
    pivtest = PIV(f"./data/animations/constant_10_100_1000/{filename}", 24, 15, 1, 0, "parabolic", False)
    X = pivtest.correlation_averaged_velocity_field[0][:, :, 0]
    Y = pivtest.correlation_averaged_velocity_field[0][:, :, 1]
    U = pivtest.correlation_averaged_velocity_field[0][:, :, 2]
    V = pivtest.correlation_averaged_velocity_field[0][:, :, 3]
    M = np.sqrt(U**2 + V**2)

    """plt.imshow(np.flip(np.flip(np.rot90(pivtest.intensity_array), axis = 1)), cmap = "Greys", aspect = "auto")
    plt.quiver(X, Y, U / M, V / M, M)
    plt.colorbar()
    plt.xlim(0, pivtest.intensity_array.shape[0])
    plt.ylim(0, pivtest.intensity_array.shape[1])
    plt.show()"""

    samples = 10000
    vels = []
    v_x = np.empty(samples, dtype = np.float64)
    v_y = np.empty(samples, dtype = np.float64)

    for i in range(samples):
        pivtest.get_resampled_correlation_averaged_velocity_field()
        v_x[i] = pivtest.resampled_correlation_averaged_velocity_field[0][:, :, 2]
        v_y[i] = pivtest.resampled_correlation_averaged_velocity_field[0][:, :, 3]

    """plt.figure(figsize = (6, 6))
    plt.title(f"{filename}")
    plt.scatter(v_x, v_y, s = 1, alpha = 0.5)
    plt.plot(U[0], V[0], marker="x", markersize=12, color="red")
    plt.axvline(5.5, c = "red")
    plt.axhline(0, c = "red")
    # plt.plot([-3, 3], [0, 0], marker = "x", markersize = 6, color = "green", ls = "None")
    plt.show()"""

    """plt.imshow(pivtest.correlation_averaged[0][0, 0, :, :])
    plt.title(f"correlation matrix for {filename}")
    plt.show()"""

    plt.figure(figsize = (6, 6))
    plt.title(f"bootstrap for {filename}")
    plt.axvline(5.5, c = "white")
    plt.axhline(0, c = "white")
    plt.axvline(U[0], c = "red")
    plt.axhline(V[0], c = "red")
    plt.hist2d(v_x, v_y, 50)
    plt.show()

# Process zebrafish
"""for filename in os.listdir("./data/processedzebra/"):
    # Plot velocity field
    pivtest = PIV(f"./data/processedzebra/{filename}", 24, 15, 10, 0.26, "fivepointgaussian", True)
    X = pivtest.correlation_averaged_velocity_field[0][:, :, 0]
    Y = pivtest.correlation_averaged_velocity_field[0][:, :, 1]
    U = pivtest.correlation_averaged_velocity_field[0][:, :, 2]
    V = pivtest.correlation_averaged_velocity_field[0][:, :, 3]
    mag = np.sqrt(U**2 + V**2)
    plt.figure(figsize = (16, 12))
    plt.imshow(np.flip(np.flip(np.rot90(pivtest.intensity_array), axis = 1)), cmap = "Greys", aspect = "auto")
    plt.quiver(X, Y, U /  mag, V / mag, mag)
    plt.clim(0, 10)
    plt.colorbar()
    plt.xlim(0, pivtest.intensity_array.shape[0])
    plt.ylim(0, pivtest.intensity_array.shape[1])
    #plt.savefig(f"./data/zebrafield/{filename}.png")
    plt.show()

# Get bootstrap
samples = 5000
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
            plt.title(f"{j}, {k}")
            #fig.add_subplot(pivtest.height, pivtest.width, k * pivtest.width + j + 1)
            #plt.axvline(0, c = "black")
            #plt.hist(vels[:, j, k].ravel(), bins = 200)
            #plt.axvline(pivtest.correlation_averaged_velocity_field[0][j, k, 2], c = "crimson")
plt.show()
print(f"Completed in {(end - start):.2f} seconds", end = "\n\n")
"""