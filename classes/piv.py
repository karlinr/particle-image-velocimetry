import numpy as np
import tifffile as tf
from scipy.optimize import curve_fit
import msl_sad_correlation as msc
import math
import matplotlib.pyplot as plt

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
        self.video = tf.imread(self.filename).astype(np.ushort)
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
        self.resampled_correlation_matrices_averaged = None

        # Run correlation averaged PIV
        #print(f"Running PIV for {self.filename} with IW:{self.iw}, SA:{self.sa}, inc:{self.inc}, threshold:{self.threshold}, pfmethod: {self.pfmethod}")
        self.run_PIV()
        #print("Complete", end = "\n\n")

    def run_PIV(self):
        self.get_image_intensity_sum()
        self.get_threshold_array()
        self.get_correlation_matrices()
        self.get_correlation_averaged()
        self.get_correlation_averaged_velocity_field()

    def get_image_intensity_sum(self):
        #print("--Getting image intensity sum...", end = " ")
        self.intensity_array = np.sum(self.video[::2], axis = 0)
        #print("complete")

    def get_threshold_array(self):
        #print("--Getting threshold array...", end = " ")
        intensity_array = np.array(
            [np.sum(self.intensity_array[j * self.inc: (j * self.inc) + (2 * self.sa + self.iw), k * self.inc: (k * self.inc) + (2 * self.sa + self.iw)]) for j in range(0, self.width) for k
             in range(0, self.height)])
        intensity_array = intensity_array - np.min(intensity_array)
        # FIX: normalisation constant is a minimum of 1 when float values could be possible
        self.threshold_array = np.array([intensity_array / np.max([np.max(intensity_array), 1]) >= self.threshold]).reshape((self.width, self.height))
        #print("complete")

    def get_correlation_matrices(self):
        #print("--Getting correlation matrices...", end = " ")
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
                        self.correlation_matrices[f // 2, j, k, :, :] = msc.sad_correlation(template.astype(np.ushort), template_to_match.astype(np.ushort))
                        # abs_diff_map = np.array([np.sum(np.abs(template_to_match[m:m + self.iw, n:n + self.iw] - template)) for m in range(0, 2 * self.sa + 1) for n in range(0, 2 * self.sa + 1)]).reshape([2 * self.sa + 1, 2 * self.sa + 1])
                        # self.correlation_matrices[f // 2, j, k, :, :] = abs_diff_map
        #print("complete")

    def get_correlation_averaged(self):
        #print("--Getting correlation averaged matrix...", end = " ")
        self.correlation_averaged = np.empty((1, *self.correlation_matrices.shape[1:5]), dtype = np.float64)
        self.correlation_averaged[0] = np.mean(self.correlation_matrices, axis = 0)
        #print("complete")

    def get_velocity_vector_from_correlation_matrix(self, _correlation_matrix):
        correlation_matrix = -_correlation_matrix + np.max(_correlation_matrix)
        peak_position = np.unravel_index(correlation_matrix.argmax(), correlation_matrix.shape)

        # methods from : https://www.aa.washington.edu/sites/aa/files/faculty/dabiri/pubs/piV.Review.Paper.final.pdf
        if self.pfmethod == "peak":
            u = peak_position[0] - (correlation_matrix.shape[0] - 1) / 2
            v = peak_position[1] - (correlation_matrix.shape[1] - 1) / 2
        elif self.pfmethod == "5pointgaussian":
            # TODO: seperate if statement so gaussian interpolation is calculated for each separately.
            if correlation_matrix.shape[0] - 1 > peak_position[0] > 0 and correlation_matrix.shape[1] - 1 > peak_position[1] > 0:
                xc = yc = math.log(correlation_matrix[peak_position[0], peak_position[1]])
                xl = math.log(correlation_matrix[peak_position[0] - 1, peak_position[1]])
                xr = math.log(correlation_matrix[peak_position[0] + 1, peak_position[1]])
                ya = math.log(correlation_matrix[peak_position[0], peak_position[1] - 1])
                yb = math.log(correlation_matrix[peak_position[0], peak_position[1] + 1])
                subpixel = [(xl - xr) / (2 * (xr - 2 * xc + xl)), (ya - yb) / (2 * (yb - 2 * yc + ya))]
                u = -(peak_position[0] - (correlation_matrix.shape[0] - 1) / 2 + subpixel[0])
                v = -(peak_position[1] - (correlation_matrix.shape[1] - 1) / 2 + subpixel[1])
            else:
                u = -(peak_position[0] - (correlation_matrix.shape[0] - 1) / 2)
                v = -(peak_position[1] - (correlation_matrix.shape[1] - 1) / 2)
        elif self.pfmethod == "9pointgaussian":
            # https://link.springer.com/content/pdf/10.1007/s00348-005-0942-3.pdf
            if correlation_matrix.shape[0] - 1 > peak_position[0] > 0 and correlation_matrix.shape[1] - 1 > peak_position[1] > 0:
                xy00 = math.log(correlation_matrix[peak_position[0] - 1, peak_position[1] - 1])
                xy01 = math.log(correlation_matrix[peak_position[0] - 1, peak_position[1]])
                xy02 = math.log(correlation_matrix[peak_position[0] - 1, peak_position[1] + 1])
                xy10 = math.log(correlation_matrix[peak_position[0], peak_position[1] - 1])
                xy11 = math.log(correlation_matrix[peak_position[0], peak_position[1]])
                xy12 = math.log(correlation_matrix[peak_position[0], peak_position[1] + 1])
                xy20 = math.log(correlation_matrix[peak_position[0] + 1, peak_position[1] - 1])
                xy21 = math.log(correlation_matrix[peak_position[0] + 1, peak_position[1]])
                xy22 = math.log(correlation_matrix[peak_position[0] + 1, peak_position[1] + 1])
                c10 = (1/6) * (-xy00 - xy01 - xy02 + xy20 + xy21 + xy22)
                c01 = (1/6) * (-xy00 - xy10 - xy20 + xy02 + xy12 + xy22)
                c11 = (1/4) * (xy00 - xy02 - xy20 + xy22)
                c20 = (1/6) * (xy00 + xy01 + xy02 - 2 * (xy10 + xy11 + xy12) + xy20 + xy21 + xy22)
                c02 = (1/6) * (xy00 + xy10 + xy20 - 2 * (xy01 + xy11 + xy21) + xy02 + xy12 + xy22)
                # c00 = (1/9) * (xy00 + 2 * xy01 - xy02 + 2 * xy01 + 5 * xy11 + 2 * xy21 - xy02 + 2 * xy12 - xy22)
                subpixel = [(c11 * c01 - 2 * c10 * c02) / (4 * c20 * c02 - c11**2), (c11 * c10 - 2 * c01 * c20) / (4 * c20 * c02 - c11**2)]
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
                return (_bg + _a * np.exp(-(((_x - _x0) ** 2 / (2 * _sigma_x ** 2)) + ((_y - _y0) ** 2 / (2 * _sigma_y ** 2))))).ravel()
            (_x, _y) = np.meshgrid(range(correlation_matrix.shape[0]), range(correlation_matrix.shape[1]))
            popt, pcov = curve_fit(gaussian2D, (_x, _y), correlation_matrix.flatten(order = "F"), maxfev = 100000)
            u = -(popt[1] - (correlation_matrix.shape[0] - 1) / 2)
            v = -(popt[2] - (correlation_matrix.shape[1] - 1) / 2)
            #print(f"{np.sqrt(pcov[1, 1]):.3f}")
            #print(f"{np.sqrt(pcov[2, 2]):.3f}")
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
        #print("--Getting velocity field for all frames...", end = " ")
        self.velocity_field = self.get_velocity_field_from_correlation_matrices(self.correlation_matrices)
        #print("complete")

    def get_correlation_averaged_velocity_field(self):
        #print("--Getting correlation averaged velocity field", end = " ")
        self.correlation_averaged_velocity_field = self.get_velocity_field_from_correlation_matrices(self.correlation_averaged)
        #print("complete")

    def get_resampled_correlation_averaged_velocity_field(self):
        indices = np.random.choice(self.correlation_matrices.shape[0], self.correlation_matrices.shape[0])
        resampled_correlation_matrices = self.correlation_matrices[indices, :, :, :]
        self.resampled_correlation_matrices_averaged = np.empty((1, self.width, self.height, 2 * self.sa + 1, 2 * self.sa + 1), dtype = np.float64)
        self.resampled_correlation_matrices_averaged[0] = np.mean(resampled_correlation_matrices, axis = 0)
        self.resampled_correlation_averaged_velocity_field = self.get_velocity_field_from_correlation_matrices(self.resampled_correlation_matrices_averaged)
