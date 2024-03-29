import numpy as np
import scipy.optimize
import tifffile as tf
import msl_sad_correlation as msc
import math
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageTransform
from scipy import ndimage

gaussiankernel = (1 / 16) * np.array([[1, 2, 1],
                                      [2, 4, 2],
                                      [1, 2, 1]])


class PIV:
    def __init__(self, _title, _iw, _sa, _inc, _threshold, _pfmethod, _pad = False, _smooth = True):
        """
        NAME
            PIV: Particle image velocimetry class

        DESCRIPTION
            Module for performing particle image velocimetry using sum of absolute differences. Implements multi-pass PIV using window translation, correlation averaging, and uncertainty quantification using correlation bootstrapping.

            Run in order:
                add_video

                set_coordinate / get_spaced_coordinates

                get_correlation_matrices

                get_correlation_averaged_velocity_field / get_velocity_field
        :param _title: String identifier used for plotting
        :param _iw: Interrogation window size in pixels
        :param _sa: Search region size in pixels
        :param _inc: Spacing between vectors
        :param _threshold: Minimum region intensity
        :param _pfmethod: Peak finding method: "peak", "5pointgaussian", "9pointgaussian", "sinc", "gaussian"
        :param _pad: Whether to pad video
        """
        # Get config variables
        self.title = _title
        self.iw = _iw
        self.sa = _sa
        self.inc = _inc
        self.threshold = _threshold
        self.pfmethod = _pfmethod
        self.pad = _pad
        self.smooth = _smooth

        # Set init variables
        self.intensity_array = None
        self.threshold_array = None
        self.correlation_matrices = None
        self.correlation_averaged = None
        self.velocity_field = None
        self.correlation_averaged_velocity_field = None
        self.resampled_correlation_averaged_velocity_field = None
        self.resampled_correlation_matrices_averaged = None
        self.samplearg = None
        self.coordinates = None
        self.xoffset = None
        self.yoffset = None
        self.video_raw = None
        self.video = None
        self.frames = None
        self.width = None
        self.height = None
        self.intensity_array_for_display = None
        self.passoffset = None

    def set(self, _iw, _sa, _inc):
        """
        Set PIV parameters.
        :param _iw:
        :param _sa:
        :param _inc:
        :return:
        """
        self.iw = _iw
        self.sa = _sa
        self.inc = _inc
        self.xoffset = int((self.video.shape[1] - ((self.width - 1) * self.inc + 2 * self.sa + self.iw)) // 2)
        self.yoffset = int((self.video.shape[2] - ((self.height - 1) * self.inc + 2 * self.sa + self.iw)) // 2)

    def set_method(self, _pfmethod):
        """
        Set peak finding method
        :param _pfmethod:
        :return:
        """
        self.pfmethod = _pfmethod

    def add_video(self, file):
        """
        Takes a list of video files and appends them to video array for PIV.
        :param file:
        :return:
        """
        if self.video_raw is None:
            if isinstance(file, str):
                self.video_raw = tf.imread(file)
            elif isinstance(file, np.ndarray) or isinstance(file, list):
                self.video_raw = tf.imread(file)
            else:
                self.video_raw = None
            if self.video_raw.ndim > 3:
                self.video_raw = np.squeeze(
                    self.video_raw.reshape((-1, self.video_raw.shape[0] * self.video_raw.shape[1], self.video_raw.shape[2], self.video_raw.shape[3])))
        else:
            if isinstance(file, str):
                video_to_add = tf.imread(file)
            elif isinstance(file, np.ndarray):
                video_to_add = file
            else:
                video_to_add = None
            if video_to_add.ndim > 3:
                video_to_add = np.squeeze(
                    video_to_add.reshape((-1, video_to_add.shape[0] * video_to_add.shape[1], video_to_add.shape[2], video_to_add.shape[3])))
            self.video_raw = np.append(self.video_raw, video_to_add, axis = 0)

        self.video = self.video_raw

        if self.pad:
            video_for_display = np.pad(self.video, [(0, 0), (int(self.sa + 0.5 * self.iw), int(self.sa + 0.5 * self.iw)),
                                                    (int(self.sa + 0.5 * self.iw), int(self.sa + 0.5 * self.iw))], mode = "minimum")
            self.video = np.pad(self.video, [(0, 0), (int(self.sa + 0.5 * self.iw), int(self.sa + 0.5 * self.iw)),
                                             (int(self.sa + 0.5 * self.iw), int(self.sa + 0.5 * self.iw))])
        else:
            video_for_display = self.video
        self.intensity_array_for_display = np.sum(video_for_display[::2], axis = 0)

        self.frames = int(self.video.shape[0])
        self.width = int(((self.video.shape[1] + self.inc) - (2 * self.sa + self.iw)) // self.inc)
        self.height = int(((self.video.shape[2] + self.inc) - (2 * self.sa + self.iw)) // self.inc)

        self.resample_reset()
        self.get_image_intensity_sum()

    def video_reset(self):
        """
        Clear video array
        :return:
        """
        self.video = None
        self.video_raw = None

    def get_spaced_coordinates(self):
        """
        Get spaced out coordinates using PIV parameters.
        :return:
        """
        # Get offset to centre vector locations
        self.width = int(((self.video.shape[1] + self.inc) - (2 * self.sa + self.iw)) // self.inc)
        self.height = int(((self.video.shape[2] + self.inc) - (2 * self.sa + self.iw)) // self.inc)
        self.xoffset = int((self.video.shape[1] - ((self.width - 1) * self.inc + 2 * self.sa + self.iw)) // 2)
        self.yoffset = int((self.video.shape[2] - ((self.height - 1) * self.inc + 2 * self.sa + self.iw)) // 2)
        self.get_threshold_array()
        self.coordinates = np.zeros((self.width, self.height, 2))
        for j in range(0, self.width):
            for k in range(0, self.height):
                # if self.threshold_array[j, k]:
                x = j * self.inc
                y = k * self.inc
                self.coordinates[j, k, :] = [x, y]
        self.passoffset = np.zeros((1, self.coordinates.shape[0], self.coordinates.shape[1], 2))

    def set_coordinate(self, x, y):
        """
        Set a coordinate for PIV analysis.
        :param x:
        :param y:
        :return:
        """
        self.xoffset = 0
        self.yoffset = 0
        self.coordinates = np.zeros((1, 1, 2))
        self.threshold_array = np.zeros((1, 1))
        self.threshold_array[0, 0] = 1
        self.coordinates[0, 0, :] = [x - self.sa - 0.5 * self.iw, y - self.sa - 0.5 * self.iw]
        self.passoffset = np.zeros((1, self.coordinates.shape[0], self.coordinates.shape[1], 2))

    def get_image_intensity_sum(self):
        """
        Gets average video intensities.
        :return: None
        """
        self.intensity_array = np.sum(self.video[self.samplearg][::2], axis = 0)

    def get_threshold_array(self):
        """
        Gets threshold array to determine where to perform PIV analysis.
        :return: None
        """
        intensity_array = np.array([np.sum(self.intensity_array[j * self.inc + self.xoffset + self.sa: (j * self.inc) + (self.sa + self.iw) + self.xoffset,
                                           k * self.inc + self.yoffset + self.sa: (k * self.inc) + (self.sa + self.iw) + self.yoffset]) for j in
                                    range(0, self.width) for k in range(0, self.height)])
        self.threshold_array = np.array(
            [(intensity_array - np.min(intensity_array)) / (np.max(intensity_array) - np.min(intensity_array)) >= self.threshold]).reshape(
            (self.width, self.height))
        return self.threshold_array

    def get_correlation_matrices(self):
        """

        :return: None
        """
        # Initialise arrays
        self.correlation_matrices = np.empty((self.frames // 2, self.coordinates.shape[0], self.coordinates.shape[1], 2 * self.sa + 1, 2 * self.sa + 1),
                                             dtype = np.float64)

        # Calculate the correlation matrices for each frame pair
        for f in range(0, self.frames, 2):
            # Get frames
            b = self.video[f]
            a = self.video[f + 1]

            # Get the absolute differences array
            for j in range(0, self.coordinates.shape[0]):
                for k in range(0, self.coordinates.shape[1]):
                    if self.threshold_array[j, k]:
                        # Get coordinates
                        tl_iw2_x = int(self.coordinates[j, k, 0] + self.xoffset - self.passoffset[0][j, k, 1])
                        tl_iw2_y = int(self.coordinates[j, k, 1] + self.yoffset - self.passoffset[0][j, k, 0])
                        br_iw2_x = int(tl_iw2_x + 2 * self.sa + self.iw)
                        br_iw2_y = int(tl_iw2_y + 2 * self.sa + self.iw)
                        tl_iw1_x = int(self.coordinates[j, k, 0] + self.xoffset + self.sa)
                        tl_iw1_y = int(self.coordinates[j, k, 1] + self.yoffset + self.sa)
                        br_iw1_x = int(tl_iw1_x + self.iw)
                        br_iw1_y = int(tl_iw1_y + self.iw)

                        # Get interrogation windows
                        template_to_match = b[tl_iw2_x:br_iw2_x, tl_iw2_y: br_iw2_y]
                        template = a[tl_iw1_x:br_iw1_x, tl_iw1_y:br_iw1_y]

                        # Calculate the absolute differences for the interrogation window
                        # self.correlation_matrices[f // 2, j, k, :, :] = ndimage.gaussian_filter(msc.sad_correlation(template.astype(np.ushort), template_to_match.astype(np.ushort)), 8)
                        self.correlation_matrices[f // 2, j, k, :, :] = msc.sad_correlation(template.astype(np.ushort), template_to_match.astype(np.ushort))

    def __get_velocity_vector_from_correlation_matrix(self, _correlation_matrix, _pfmethod = None):
        """

        :param _correlation_matrix:
        :param _pfmethod: peak-finding method: "peak", "5pointgaussian", "9pointgaussian", "sinc", "gaussian"
        :return:
        """
        if self.smooth:
            #correlation_matrix = ndimage.convolve(-_correlation_matrix + np.max(_correlation_matrix), gaussiankernel)
            correlation_matrix = ndimage.gaussian_filter(-_correlation_matrix + np.max(_correlation_matrix), 3, mode = "wrap")
            """plt.imshow(correlation_matrix)
            plt.show()"""
        else:
            correlation_matrix = -_correlation_matrix + np.max(_correlation_matrix)
        peak_position = np.unravel_index(correlation_matrix.argmax(), correlation_matrix.shape, order = "C")

        # methods from : https://www.aa.washington.edu/sites/aa/files/faculty/dabiri/pubs/piV.Review.Paper.final.pdf
        if _pfmethod is None:
            pfmethod = self.pfmethod
        else:
            pfmethod = _pfmethod
        if pfmethod == "5pointgaussian":
            # TODO: seperate if statement so gaussian interpolation is calculated for each separately.
            try:
                if correlation_matrix.shape[0] - 1 > peak_position[0] > 0 and correlation_matrix.shape[1] - 1 > peak_position[1] > 0:
                    xc = yc = math.log(correlation_matrix[peak_position[0], peak_position[1]])
                    xl = math.log(correlation_matrix[peak_position[0] - 1, peak_position[1]])
                    xr = math.log(correlation_matrix[peak_position[0] + 1, peak_position[1]])
                    ya = math.log(correlation_matrix[peak_position[0], peak_position[1] - 1])
                    yb = math.log(correlation_matrix[peak_position[0], peak_position[1] + 1])
                    subpixel = [(xl - xr) / (2 * (xr - 2 * xc + xl)), (ya - yb) / (2 * (yb - 2 * yc + ya))]
                    u = -(peak_position[0] + subpixel[0] - (correlation_matrix.shape[0] - 1) / 2)
                    v = -(peak_position[1] + subpixel[1] - (correlation_matrix.shape[1] - 1) / 2)
                else:
                    # print("Falling back to peak fitting method, try increasing window search area")
                    return self.__get_velocity_vector_from_correlation_matrix(_correlation_matrix, "peak")
            except ValueError:
                print("Math error.")
                return self.__get_velocity_vector_from_correlation_matrix(_correlation_matrix, "sinc")
        elif pfmethod == "peak":
            u = -(peak_position[0] - (correlation_matrix.shape[0] - 1) / 2)
            v = -(peak_position[1] - (correlation_matrix.shape[1] - 1) / 2)
        elif pfmethod == "parabolic":
            if correlation_matrix.shape[0] - 1 > peak_position[0] > 0 and correlation_matrix.shape[1] - 1 > peak_position[1] > 0:
                xc = yc = correlation_matrix[peak_position[0], peak_position[1]]
                xl = correlation_matrix[peak_position[0] - 1, peak_position[1]]
                xr = correlation_matrix[peak_position[0] + 1, peak_position[1]]
                ya = correlation_matrix[peak_position[0], peak_position[1] - 1]
                yb = correlation_matrix[peak_position[0], peak_position[1] + 1]
                subpixel = [(xl - xr) / (2 * (xl + xr - 2 * xc)), (ya - yb) / (2 * (ya + yb - 2 * yc))]
                u = -(peak_position[0] + subpixel[0] - (correlation_matrix.shape[0] - 1) / 2)
                v = -(peak_position[1] + subpixel[1] - (correlation_matrix.shape[1] - 1) / 2)
            else:
                # print("Falling back to peak fitting method, try increasing window search area")
                return self.__get_velocity_vector_from_correlation_matrix(_correlation_matrix, "peak")
        elif pfmethod == "9pointgaussian":
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
                c10 = (1 / 6) * (-xy00 - xy01 - xy02 + xy20 + xy21 + xy22)
                c01 = (1 / 6) * (-xy00 - xy10 - xy20 + xy02 + xy12 + xy22)
                c11 = (1 / 4) * (xy00 - xy02 - xy20 + xy22)
                c20 = (1 / 6) * (xy00 + xy01 + xy02 - 2 * (xy10 + xy11 + xy12) + xy20 + xy21 + xy22)
                c02 = (1 / 6) * (xy00 + xy10 + xy20 - 2 * (xy01 + xy11 + xy21) + xy02 + xy12 + xy22)
                # c00 = (1/9) * (xy00 + 2 * xy01 - xy02 + 2 * xy01 + 5 * xy11 + 2 * xy21 - xy02 + 2 * xy12 - xy22)
                subpixel = [(c11 * c01 - 2 * c10 * c02) / (4 * c20 * c02 - c11 ** 2), (c11 * c10 - 2 * c01 * c20) / (4 * c20 * c02 - c11 ** 2)]
                u = -(peak_position[0] - (correlation_matrix.shape[0] - 1) / 2 + subpixel[0])
                v = -(peak_position[1] - (correlation_matrix.shape[1] - 1) / 2 + subpixel[1])
            else:
                # print("Falling back to peak fitting method, try increasing window search area.")
                return self.__get_velocity_vector_from_correlation_matrix(_correlation_matrix, "peak")
        elif pfmethod == "sinc":
            # https://www.ipol.im/pub/art/2011/g_lmii/article.pdf
            # Slow and buggy - do not use
            if correlation_matrix.shape[0] - 1 > peak_position[0] > 0 and correlation_matrix.shape[1] - 1 > peak_position[1] > 0:
                def get_sinc_interpolation(x):
                    (m, n) = np.meshgrid(range(correlation_matrix.shape[0]), range(correlation_matrix.shape[1]))
                    return -np.sum(correlation_matrix[m, n] * np.sinc(x[0] - m) * np.sinc(x[1] - n))

                res = scipy.optimize.minimize(get_sinc_interpolation, x0 = [peak_position[0], peak_position[1]],
                                              bounds = ((peak_position[0] - 1, peak_position[0] + 1), (peak_position[1] - 1, peak_position[1] + 1)))
                u = -(res.x[0] - (correlation_matrix.shape[0] - 1) / 2)
                v = -(res.x[1] - (correlation_matrix.shape[1] - 1) / 2)
            else:
                # print("Falling back to peak fitting method, try increasing window search area.")
                return self.__get_velocity_vector_from_correlation_matrix(_correlation_matrix, "peak")
        elif pfmethod == "gaussian":
            # Slow and buggy - do not use
            def gaussian2D(_xy, _a, _x0, _y0, _sigma_x, _sigma_y, _bg):
                return (_bg + _a * np.exp(-(((_x - _x0) ** 2 / (2 * _sigma_x ** 2)) + ((_y - _y0) ** 2 / (2 * _sigma_y ** 2))))).ravel()

            try:
                (_x, _y) = np.meshgrid(range(correlation_matrix.shape[0]), range(correlation_matrix.shape[1]))
                popt, pcov = scipy.optimize.curve_fit(gaussian2D, (_x, _y), correlation_matrix.flatten(order = "F"), maxfev = 100000)
                u = -(popt[1] - (correlation_matrix.shape[0] - 1) / 2)
                v = -(popt[2] - (correlation_matrix.shape[1] - 1) / 2)
            except RuntimeError:
                # print("Falling back to 9pointgaussian fitting method, curve fit unsuccessful.")
                return self.__get_velocity_vector_from_correlation_matrix(_correlation_matrix, "9pointgaussian")
        else:
            raise ValueError("Invalid peak fitting method.")
        return v, u

    def get_averaged_peak_correlation_amplitude(self):
        return np.max(-self.correlation_averaged + np.max(self.correlation_averaged))

    def get_peak_correlation_amplitude(self):
        correlation_matrices = self.correlation_matrices[self.samplearg, 0, 0, :, :]
        peaks = []
        for f in range(0, correlation_matrices.shape[0]):
            correlation_matrix = -correlation_matrices[f] + np.max(correlation_matrices[f])
            peak = np.max(correlation_matrix)
            peaks.append(peak)
        return peaks

    # SNR Metrics
    # Only work with single coordinate
    # https://arxiv.org/ftp/arxiv/papers/1405/1405.3023.pdf
    def get_averaged_peak_to_root_mean_square_ratio(self):
        correlation_matrix = -self.correlation_averaged[0, 0, 0, :, :] + np.max(self.correlation_averaged[0, 0, 0, :, :])
        peak = np.max(correlation_matrix)
        halfpoints = np.where(correlation_matrix <= peak / 2)
        peak_rms = np.sqrt((1 / np.count_nonzero(halfpoints)) * np.sum(correlation_matrix[halfpoints] ** 2))
        return peak ** 2 / peak_rms ** 2

    def get_peak_to_root_mean_square_ratio(self):
        prmsr = []
        correlation_matrices = self.correlation_matrices[self.samplearg, 0, 0, :, :]
        for f in range(0, correlation_matrices.shape[0]):
            correlation_matrix = -correlation_matrices[f] + np.max(correlation_matrices[f])
            peak = np.max(correlation_matrix)
            halfpoints = np.where(correlation_matrix <= peak / 2)
            peak_rms = ((1 / np.count_nonzero(halfpoints)) * np.sum(correlation_matrix[halfpoints] ** 2)) ** 0.5
            prmsr.append(peak ** 2 / peak_rms ** 2)
        return prmsr

    def __get_velocity_field_from_correlation_matrices(self, correlation_matrix):
        """

        :param correlation_matrix:
        :return: velocity_field
        """
        # velocity_field = np.zeros((self.frames // 2, self.coordinates.shape[0], self.coordinates.shape[1], 4))
        velocity_field = np.zeros((correlation_matrix.shape[0], self.coordinates.shape[0], self.coordinates.shape[1], 2))
        for f in range(0, correlation_matrix.shape[0]):
            for j in range(0, self.coordinates.shape[0]):
                for k in range(0, self.coordinates.shape[1]):
                    if self.threshold_array[j, k]:
                        velocity_field[f, j, k, :] = [*self.__get_velocity_vector_from_correlation_matrix(correlation_matrix[f, j, k])]
        return velocity_field + self.passoffset

    def do_pass(self):
        self.passoffset = np.round(self.correlation_averaged_velocity_field)

    def resample(self, sample_size = None):
        if sample_size is None:
            sample_size = self.video.shape[0] // 2
        self.samplearg = np.random.choice(self.video.shape[0] // 2, sample_size)

    def resample_specific(self, args, intensity_array = False):
        self.samplearg = args
        if intensity_array:
            self.get_image_intensity_sum()
            self.get_threshold_array()

    def resample_from_array(self, arr, sample_size = None):
        if sample_size is None:
            sample_size = len(arr)
        self.samplearg = np.random.choice(arr, sample_size)

    def resample_reset(self):
        self.samplearg = np.arange(self.video.shape[0] // 2)

    def get_uncertainty(self, iterations):
        distribution_x = []
        distribution_y = []
        for _ in range(iterations):
            self.resample()
            self.get_correlation_averaged_velocity_field()
            distribution_x.append(self.x_velocity_averaged())
            distribution_y.append(self.y_velocity_averaged())
        return np.std(distribution_x, axis = 0), np.std(distribution_y, axis = 0)

    def get_distribution(self, iterations):
        distribution_x = []
        distribution_y = []
        for _ in range(iterations):
            distribution_x.append(self.x_velocity_averaged())
            distribution_y.append(self.y_velocity_averaged())
        return distribution_x, distribution_y

    def get_velocity_field(self):
        """

        :return:
        """
        self.velocity_field = self.__get_velocity_field_from_correlation_matrices(self.correlation_matrices)

    def get_correlation_averaged_velocity_field(self):
        """

        :return:
        """
        self.correlation_averaged = np.empty((1, *self.correlation_matrices[self.samplearg].shape[1:5]), dtype = np.float64)
        self.correlation_averaged[0] = np.mean(self.correlation_matrices[self.samplearg], axis = 0)
        self.correlation_averaged_velocity_field = self.__get_velocity_field_from_correlation_matrices(self.correlation_averaged)

    def window_deform(self, amount = 1):
        """
        Applies a window deformation to odd frames based upon current measured flow field.
        :return: None
        """
        xcoords = self.xcoords().astype(int)
        ycoords = self.ycoords().astype(int)
        flow_x = -self.x_velocity_averaged() * amount
        flow_y = -self.y_velocity_averaged() * amount
        mesh = []
        for j in range(xcoords.shape[0] - 1):
            for k in range(ycoords.shape[1] - 1):
                mesh.append([(xcoords[j, k], ycoords[j, k], xcoords[j + 1, k + 1], ycoords[j + 1, k + 1]),
                             [xcoords[j, k] + flow_x[j, k], ycoords[j, k] + flow_y[j, k],
                              xcoords[j + 1, k] + flow_x[j + 1, k], ycoords[j + 1, k] + flow_y[j + 1, k],
                              xcoords[j + 1, k + 1] + flow_x[j + 1, k + 1], ycoords[j + 1, k + 1] + flow_y[j + 1, k + 1],
                              xcoords[j, k + 1] + flow_x[j, k + 1], ycoords[j, k + 1] + flow_y[j, k + 1]]])

        for i in range(self.video.shape[0] // 2):
            image_deformed = Image.fromarray(self.video[i * 2, :, :], mode = "I;16")
            image_deformed = np.array(image_deformed.transform(image_deformed.size, PIL.Image.MESH, mesh))
            self.video[i * 2, :, :] = image_deformed

    # Return methods
    def x_velocity_averaged(self):
        return self.correlation_averaged_velocity_field[0][:, :, 0]

    def y_velocity_averaged(self):
        return self.correlation_averaged_velocity_field[0][:, :, 1]

    def x_velocity(self):
        return self.velocity_field[:, :, :, 0]

    def y_velocity(self):
        return self.velocity_field[:, :, :, 1]

    def velocity_magnitude_averaged(self):
        return np.sqrt(self.x_velocity_averaged()[:, :] ** 2 + self.y_velocity_averaged()[:, :] ** 2)

    def velocity_angle_averaged(self):
        return np.arctan2(self.y_velocity_averaged()[:, :], self.x_velocity_averaged()[:, :])

    def xcoords(self):
        return self.coordinates[:, :, 1] + self.sa + 0.5 * self.iw + self.yoffset

    def ycoords(self):
        return self.coordinates[:, :, 0] + self.sa + 0.5 * self.iw + self.xoffset

    # Drawing methods
    def plot_flow_field(self, savelocation = None, frame = None):
        # plt.figure(figsize = (12, 7))
        if frame is None:
            plt.title(f"{self.title}")
            # U = self.x_velocity_averaged()[:, :]
            # V = self.y_velocity_averaged()[:, :]
            U = self.x_velocity_averaged()[:, :]
            V = self.y_velocity_averaged()[:, :]
        else:
            plt.title(f"{self.title}\n Frame : {frame}")
            U = self.x_velocity()[frame, :, :]
            V = self.y_velocity()[frame, :, :]
        mag = np.sqrt(U ** 2 + V ** 2)
        plt.imshow(self.intensity_array_for_display, cmap = "gray", aspect = "auto")
        plt.quiver(self.xcoords(), self.ycoords(), U / mag, V / mag, mag, angles = "xy")
        plt.colorbar()
        # plt.imshow(self.intensity_array_for_display, origin = "lower", cmap = "gray")
        # plt.clim(0, 24)
        # plt.ylim(0, self.intensity_array.shape[0])
        # plt.xlim(0, self.intensity_array.shape[1])
        # plt.gca().invert_yaxis()
        if savelocation is not None:
            plt.savefig(savelocation)
        plt.show()
        plt.close()

    def begin_draw(self):
        plt.figure(figsize = (3, 3))
        #plt.figure(figsize = (6.2, 3.4))

    def draw_iw(self):
        for j in range(0, self.coordinates.shape[0]):
            for k in range(0, self.coordinates.shape[1]):
                if self.threshold_array[j, k]:
                    tl_iw1_x = int(self.coordinates[j, k, 0] + self.xoffset + self.sa)
                    tl_iw1_y = int(self.coordinates[j, k, 1] + self.yoffset + self.sa)

                    rectangle = plt.Rectangle((tl_iw1_y, tl_iw1_x), self.iw, self.iw, fc = 'none', ec = "red")
                    plt.gca().add_patch(rectangle)

    def draw_sa(self):
        for j in range(0, self.coordinates.shape[0]):
            for k in range(0, self.coordinates.shape[1]):
                if self.threshold_array[j, k]:
                    tl_iw2_x = int(self.coordinates[j, k, 0] + self.xoffset + self.passoffset[0][j, k, 0])
                    tl_iw2_y = int(self.coordinates[j, k, 1] + self.yoffset + self.passoffset[0][j, k, 1])

                    rectangle = plt.Rectangle((tl_iw2_y, tl_iw2_x), self.iw + 2 * self.sa, self.iw + 2 * self.sa, fc = 'none', ec = "green")
                    plt.gca().add_patch(rectangle)

    def draw_intensity(self):
        plt.imshow(self.intensity_array_for_display, cmap = "gray", aspect = "equal", interpolation = "None")

    def draw_flow_field(self, frame = None):
        if frame is None:
            plt.title(f"{self.title}")
            # U = self.x_velocity_averaged()[:, :]
            # V = self.y_velocity_averaged()[:, :]
            U = self.x_velocity_averaged()[:, :]
            V = self.y_velocity_averaged()[:, :]
        else:
            plt.title(f"{self.title}\n Frame : {frame}")
            U = self.x_velocity()[frame, :, :]
            V = self.y_velocity()[frame, :, :]
        mag = np.sqrt(U ** 2 + V ** 2)**0.5
        plt.quiver(self.xcoords(), self.ycoords(), U / mag, V / mag, mag, angles = "xy")
        plt.colorbar()
        plt.xlim(self.sa + self.iw, self.intensity_array_for_display.shape[1] - self.sa - self.iw)
        plt.ylim(self.sa + self.iw, self.intensity_array_for_display.shape[0] - self.sa - self.iw)
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def draw_averaged_correlation_matrix(self):
        plt.imshow(self.correlation_averaged[0, 0, 0])

    def end_draw(self, draw = True, save = False):
        if save != False:
            plt.savefig(save)
        if draw:
            plt.show()
