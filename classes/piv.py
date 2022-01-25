import numpy as np
import scipy.optimize
import tifffile as tf
import msl_sad_correlation as msc
import math
import matplotlib.pyplot as plt


class PIV:
    def __init__(self, _filename, _iw, _sa, _inc, _threshold, _pfmethod, _pad = False):
        """
        Creates a particle image velocimetry object.
        :param _filename: A tif video
        :param _iw: Inner interrogation window
        :param _sa: Search area for larger interrogation window
        :param _inc: Increment for velocity field
        :param _threshold: Minimum particle threshold
        :param _pfmethod: peak-finding method: "peak", "5pointgaussian", "9pointgaussian", "sinc", "gaussian"
        :param _pad: Whether to pad the input tif video
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

        self.resample_reset()
        self.__get_image_intensity_sum()

    def get_spaced_coordinates(self):
        # Get offset to centre vector locations
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

    def set_coordinate(self, x, y):
        self.xoffset = 0
        self.yoffset = 0
        self.coordinates = np.zeros((1, 1, 2))
        self.threshold_array = np.zeros((1, 1))
        self.threshold_array[0, 0] = 1
        self.coordinates[0, 0, :] = [x - self.sa - 0.5 * self.iw, y - self.sa - 0.5 * self.iw]

    def __get_image_intensity_sum(self):
        """

        :return: None
        """
        self.intensity_array = np.sum(self.video[::2], axis = 0)

    def get_threshold_array(self):
        """

        :return: None
        """
        intensity_array = np.array(
            [np.sum(self.intensity_array[j * self.inc + self.xoffset: (j * self.inc) + (2 * self.sa + self.iw) + self.xoffset,
                    k * self.inc + self.yoffset: (k * self.inc) + (2 * self.sa + self.iw) + self.yoffset]) for j
             in range(0, self.width) for k
             in range(0, self.height)])
        intensity_array = intensity_array - np.min(intensity_array)
        # TODO: fix : normalisation constant is a minimum of 1 when float values could be possible
        self.threshold_array = np.array([intensity_array / np.max([np.max(intensity_array), 1]) >= self.threshold]).reshape((self.width, self.height))

    def get_correlation_matrices(self):
        """

        :return: None
        """
        # Initialise arrays
        self.correlation_matrices = np.empty((self.frames // 2, self.coordinates.shape[0], self.coordinates.shape[1], 2 * self.sa + 1, 2 * self.sa + 1), dtype = np.uintc)

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
                        tl_iw2_x = int(self.coordinates[j, k, 0] + self.xoffset)
                        tl_iw2_y = int(self.coordinates[j, k, 1] + self.yoffset)
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

    def __get_velocity_vector_from_correlation_matrix(self, _correlation_matrix, _pfmethod = None):
        """

        :param _correlation_matrix:
        :param _pfmethod: peak-finding method: "peak", "5pointgaussian", "9pointgaussian", "sinc", "gaussian"
        :return:
        """
        correlation_matrix = -_correlation_matrix + np.max(_correlation_matrix)
        peak_position = np.unravel_index(correlation_matrix.argmax(), correlation_matrix.shape)

        # methods from : https://www.aa.washington.edu/sites/aa/files/faculty/dabiri/pubs/piV.Review.Paper.final.pdf
        if _pfmethod is None:
            pfmethod = self.pfmethod
        else:
            pfmethod = _pfmethod
        if pfmethod == "peak":
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
                u = -(peak_position[0] - (correlation_matrix.shape[0] - 1) / 2 + subpixel[0])
                v = -(peak_position[1] - (correlation_matrix.shape[1] - 1) / 2 + subpixel[1])
            else:
                #print("Falling back to peak fitting method, try increasing window search area")
                return self.__get_velocity_vector_from_correlation_matrix(_correlation_matrix, "peak")
        elif pfmethod == "5pointgaussian":
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
                #print("Falling back to peak fitting method, try increasing window search area")
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
                #print("Falling back to peak fitting method, try increasing window search area.")
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
                #print("Falling back to peak fitting method, try increasing window search area.")
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
                #print("Falling back to 9pointgaussian fitting method, curve fit unsuccessful.")
                return self.__get_velocity_vector_from_correlation_matrix(_correlation_matrix, "9pointgaussian")
        else:
            raise ValueError("Invalid peak fitting method.")
        return u, v

    def __get_velocity_field_from_correlation_matrices(self, correlation_matrix):
        """

        :param correlation_matrix:
        :return: velocity_field
        """
        velocity_field = np.zeros((self.frames // 2, self.coordinates.shape[0], self.coordinates.shape[1], 4))
        for f in range(0, correlation_matrix.shape[0]):
            for j in range(0, self.coordinates.shape[0]):
                for k in range(0, self.coordinates.shape[1]):
                    if self.threshold_array[j, k]:
                        velocity_field[f, j, k, :] = [self.coordinates[j, k, 0], self.coordinates[j, k, 1], *self.__get_velocity_vector_from_correlation_matrix(correlation_matrix[f, j, k])]
        return velocity_field

    def resample(self, sample_size = None):
        if sample_size is None:
            sample_size = self.video.shape[0] // 2
        self.samplearg = np.random.choice(self.video.shape[0] // 2, sample_size)

    def resample_specific(self, args):
        self.samplearg = args

    def resample_from_array(self, arr, sample_size = None):
        if sample_size is None:
            sample_size = len(arr)
        self.samplearg = np.random.choice(arr, sample_size)

    def resample_reset(self):
        self.samplearg = np.arange(self.video.shape[0] // 2)

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

    # Todo : Add error checking
    # Return methods
    def x_velocity_averaged(self):
        return self.correlation_averaged_velocity_field[0][:, :, 2]

    def y_velocity_averaged(self):
        return self.correlation_averaged_velocity_field[0][:, :, 3]

    def x_velocity(self):
        return self.velocity_field[:, :, :, 2]

    def y_velocity(self):
        return self.velocity_field[:][:, :, 3]

    def velocity_magnitude_averaged(self):
        return np.sqrt(self.x_velocity_averaged()[:, :] ** 2 + self.y_velocity_averaged()[:, :] ** 2)

    def velocity_angle_averaged(self):
        return np.arctan2(self.y_velocity_averaged()[:, :], self.x_velocity_averaged()[:, :])

    def xcoords(self):
        return self.coordinates[:, :, 0] + self.sa + 0.5 * self.iw

    def ycoords(self):
        return self.coordinates[:, :, 1] + self.sa + 0.5 * self.iw

    def plot_flow_field(self, savelocation = None, frame = None):
        plt.figure()
        #plt.figure(figsize = (12, 7))
        if frame is None:
            plt.title(f"{self.filename}\n Averaged")
            U = self.x_velocity_averaged()[:, :]
            V = self.y_velocity_averaged()[:, :]
        else:
            plt.title(f"{self.filename}\n Frame : {frame}")
            U = self.x_velocity(frame)[:, :]
            V = self.y_velocity(frame)[:, :]
        mag = np.sqrt(U ** 2 + V ** 2)
        plt.imshow(np.flip(np.flip(np.rot90(self.intensity_array_for_display), axis = 1)), cmap = "gray", aspect = "auto")
        plt.quiver(self.xcoords(), self.ycoords(), U / mag, V / mag, mag, angles = "xy")
        # plt.clim(0, 24)
        plt.colorbar()
        plt.xlim(0, self.intensity_array.shape[0])
        plt.ylim(0, self.intensity_array.shape[1])
        plt.gca().invert_yaxis()
        if savelocation is not None:
            plt.savefig(savelocation)
        plt.show()
        plt.clf()
