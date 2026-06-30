import numpy as np
import colour

xyz_to_rgb = np.array([[2.3706743, -0.9000405, -0.4706338],
                       [-0.5138850, 1.4253036,  0.0885814],
                       [ 0.0052982, -0.0146949,  1.0093968]])

rgb_to_xyz = np.array([[0.4887180,  0.3106803,  0.2006017],
                       [0.1762044,  0.8129847,  0.0108109],
                       [0.0000000,  0.0102048,  0.9897952]])


def interpolate_spectrum(x_values, y_values, new_x):
    """
    Sprague (1880) method is recommended by the CIE for interpolating
    functions having a uniformly spaced independent variable.
    :param x_values: Independent 𝑥 variable values corresponding with 𝑦 variable.
    :param y_values: Dependent and already known 𝑦 variable values to interpolate.
    :param new_x: new independent 𝑥 variable values
    :return: new y variable values
    """
    f = colour.SpragueInterpolator(x_values, y_values)
    return f(new_x)


def lut(t):
    """"" Gamma correction """
    return t ** 2.2 / (255 ** 2.2)


def get_stimulus(tristimulus_color, spd):
    """ The formation of an additive three-component stimulus """
    return lut(tristimulus_color[0]) * spd[0] + lut(tristimulus_color[1]) * spd[1] + lut(tristimulus_color[2]) * spd[2]

################################################################################
#              Class for Working with observer/sensor sensitivities            #
################################################################################

class Color_perseption(object):
    def __init__(self, wl=np.arange(380, 781, 1)):
        self.type = "Observer"  # Observer / Camera
        self.wl = wl
        self.sensitivity = np.empty

    def load_CMFs(self, perseption_type):
        self.sensitivity = perseption_type
        self.sensitivity /= self.sensitivity.mean()
        # print(self.sensitivity)
        self.type = "Observer"

    def load_SSS(self, perseption_type):
        self.sensitivity = perseption_type
        self.type = "Camera"

    def set_white_point(self, white_point):
        self.white_point = white_point

    def set_white_illum(self, white_illum):
        if (self.type == "Observer"):
            self.white_illum = np.ones(len(self.wl))
            XYZ = self.calculate_xyz_from_spectrum(white_illum)
            XYZ /= XYZ[1]
            self.white_point = colour.XYZ_to_xy(XYZ)
            self.white_illum = white_illum

            """ sensor white balance """
        else:
            # illuminant.values = illuminant.values / np.max(illuminant.values)
            self.white_illum = white_illum
            coef = 100 * self.sensitivity.dot(np.ones(len(white_illum))) / len(self.sensitivity)
            self.sensitivity = self.sensitivity / coef[:, None]
            # self.sensitivity = (self.sensitivity.T / self.sensitivity.mean(axis = 1)).T

    def calculate_xyz_from_spectrum(self, stimul):
        k = 1 / (np.sum(self.sensitivity[1] * self.white_illum))
        X_p = stimul * self.sensitivity[0] * self.white_illum
        Y_p = stimul * self.sensitivity[1] * self.white_illum
        Z_p = stimul * self.sensitivity[2] * self.white_illum
        XYZ = k * np.sum(np.array([X_p, Y_p, Z_p]), axis=-1)

        return XYZ

    def calculate_rgb_from_spectrum(self, stimul, dw=1):
        if (self.type == "Observer"):
            XYZ = self.calculate_xyz_from_spectrum(stimul)
            rgb = xyz_to_rgb.dot(XYZ)
            # rgb = np.round(np.array(rgb * 255)).astype(int)

            return rgb

        elif (self.type == "Camera"):
            k = 1 / (np.sum(self.sensitivity[1] * self.white_illum))
            r = stimul * self.sensitivity[0] * self.white_illum
            g = stimul * self.sensitivity[1] * self.white_illum
            b = stimul * self.sensitivity[2] * self.white_illum
            # rgb = 1 / 3. * np.sum(np.array([r, g, b]), axis=-1)
            rgb = k * np.sum(np.array([r, g, b]), axis=-1)

            return rgb
