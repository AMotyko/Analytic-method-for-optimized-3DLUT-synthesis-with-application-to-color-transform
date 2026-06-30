import colour
import numpy as np

from reflectance_reconstruction import reflectance_reconstruction_prep, reflectance_reconstruction
from dataset import Dataset
from cmf import *
from colour.models import xy_to_xyY, xyY_to_XYZ
from colour.adaptation.vonkries import matrix_chromatic_adaptation_VonKries
from colour.colorimetry import intermediate_lightness_function_CIE1976
from colour.utilities import (from_range_100, tsplit, tstack)
import pandas as pd
import scipy


prolab_transform_M = np.array([[79.4725, 486.6610, 153.7311, 0],
                               [649.9038, -595.4477, -20.4498, 0],
                               [50.8625, 194.9377, -223.4334, 0],
                               [0.7947, 3.8666, 1.5373, 1]])

def XYZ_to_Prolab(xyz):
    T_h = prolab_transform_M.dot(np.append(xyz, 1))
    T_c = np.delete(T_h / T_h[3], 3)
    return T_c

class DataGenerator(object):
    def __init__(self):
        self.available_parameters = {
            'color_difference_model': ['CIEDE2000', 'Prolab', 'L2 in RGB']
        }
        self.wl = np.arange(380, 781, 1)
        self.CMFs = np.array
        self.base_white_illum = 'D65'
        self.base_white_illum_spectrum = np.array
        self.reflectances = np.array
        self.SSS = np.array
        self.white_illum = np.array
        self.color_difference_model = 'Prolab'
        self.XYZ_ref = [1, 1, 1]

        self.obj_observer = Color_perseption
        self.obj_sensor = Color_perseption

        self.xyz_observer_arr = np.array
        self.rgb_sensor_arr = np.array
        self.M_CAT = np.array

    def initialize(self):
        self.obj_observer = Color_perseption(self.wl)
        self.obj_observer.load_CMFs(self.CMFs)
        self.obj_observer.set_white_illum(self.base_white_illum_spectrum)
        self.XYZ_ref = self.obj_observer.calculate_xyz_from_spectrum(np.ones(len(self.wl)))
        # print("ref_XYZ: ", self.ref_XYZ)
        self.obj_observer.set_white_illum(self.white_illum)  # here we also found white point using observer CMFs
        #
        self.obj_sensor = Color_perseption(self.wl)
        self.obj_sensor.load_SSS(self.SSS)
        self.obj_sensor.set_white_illum(self.white_illum)
        self.obj_sensor.set_white_point(
            self.obj_observer.white_point)  # here we set white point found for observer CMFs

        chromatic_adaptation_transform = "Bradford"
        self.M_CAT = matrix_chromatic_adaptation_VonKries(
            xyY_to_XYZ(xy_to_xyY(self.obj_observer.white_point)),
            self.XYZ_ref,
            transform=chromatic_adaptation_transform)
        # D = 0.3
        # self.M_CAT = D * self.M_CAT + (1 - D) * np.eye(3)
        self.get_xyz_and_rgb_input_values()

    def set_spectral_data(self, reflectances, cmf, sss, white_illuminant, illum_name, wl_step=1, print_info=True):
        if (print_info == True):
            print("Spectral data succesfully loaded.")
            print("Reflectances:                 from", reflectances['wl'].iloc[0], 'nm to',
                  reflectances['wl'].iloc[-1], 'nm')
            print("Color matching functions:     from", cmf['wl'].iloc[0], 'nm to', cmf['wl'].iloc[-1], 'nm')
            print("Spectral sensor sensitivity:  from", sss['wl'].iloc[0], 'nm to', sss['wl'].iloc[-1], 'nm')
            print("White illuminant:             from", white_illuminant['wl'].iloc[0], 'nm to',
                  white_illuminant['wl'].iloc[-1], 'nm')

        self.wl = pd.merge(
            pd.merge(pd.merge(reflectances['wl'], sss['wl'], how='inner'), white_illuminant['wl'], how='inner'),
            cmf['wl'], how='inner').values[:, 0]
        l_reflectances = reflectances.loc[reflectances['wl'].isin(self.wl)]
        l_cmf = cmf.loc[cmf['wl'].isin(self.wl)]
        l_sss = sss.loc[sss['wl'].isin(self.wl)]
        l_white_illuminant = white_illuminant.loc[white_illuminant['wl'].isin(self.wl)]
        dataset = Dataset()
        l_base_white_illum_spectrum = dataset.illuminant.cie.std()
        l_base_white_illum_spectrum = l_base_white_illum_spectrum.loc[l_base_white_illum_spectrum['wl'].isin(self.wl)]

        if (wl_step != reflectances['wl'].iloc[1] - reflectances['wl'].iloc[0]):
            new_wl = np.arange(self.wl[0], self.wl[-1] + wl_step, wl_step)
            l_reflectances = l_reflectances.apply(lambda x: interpolate_spectrum(self.wl, x, new_wl))

        if (wl_step != l_white_illuminant['wl'].iloc[1] - l_white_illuminant['wl'].iloc[0]):
            new_wl = np.arange(self.wl[0], self.wl[-1] + wl_step, wl_step)
            l_white_illuminant = l_white_illuminant.apply(lambda x: interpolate_spectrum(self.wl, x, new_wl))

        if (wl_step != 1):
            new_wl = np.arange(self.wl[0], self.wl[-1] + wl_step, wl_step)
            l_cmf = l_cmf.apply(lambda x: interpolate_spectrum(self.wl, x, new_wl))
            l_sss = l_sss.apply(lambda x: interpolate_spectrum(self.wl, x, new_wl))
            l_base_white_illum_spectrum = l_base_white_illum_spectrum.apply(
                lambda x: interpolate_spectrum(self.wl, x, new_wl))
            self.wl = new_wl

        if (print_info == True):
            print("The final spectral range is   from", self.wl[0], 'nm to', self.wl[-1], 'nm with step', wl_step)

        self.reflectances = np.transpose(l_reflectances.drop(columns=['wl']).values)
        self.CMFs = np.transpose(l_cmf.drop(columns=['wl']).values)
        self.SSS = np.transpose(l_sss.drop(columns=['wl']).values)
        self.white_illum = l_white_illuminant[illum_name].values
        self.base_white_illum_spectrum = l_base_white_illum_spectrum[self.base_white_illum].values

        self.initialize()

        # if (normalize_illum == True):
        #     self.white_illum *= (100 / self.white_illum.mean())


    def set_color_difference_model(self, model):
        if model in self.available_parameters['color_difference_model']:
            self.color_difference_model = model
            self.base_white_illum = "D50" if model == "CIEDE2000" else "D65"
        else:
            raise ValueError(
                "Parameter '" + model + "' for method 'set_color_difference_model' not found. Please, use one of "
                                        "this parameters: " + str(self.available_parameters['color_difference_model']))

    def XYZ_to_Lab(self, XYZ):
        X, Y, Z = tsplit(XYZ)
        X_n, Y_n, Z_n = tsplit(self.XYZ_ref)
        f_X_X_n = intermediate_lightness_function_CIE1976(X, X_n)
        f_Y_Y_n = intermediate_lightness_function_CIE1976(Y, Y_n)
        f_Z_Z_n = intermediate_lightness_function_CIE1976(Z, Z_n)

        L = 116 * f_Y_Y_n - 16
        a = 500 * (f_X_X_n - f_Y_Y_n)
        b = 200 * (f_Y_Y_n - f_Z_Z_n)

        Lab = from_range_100(tstack([L, a, b]))
        return Lab

    def calc_color_difference_CIEDE2000(self, color1_xyz, color2_xyz):
        """ Calculation of the color difference in XYZ color space using CIEDE2000 metric """
        color1_lab = self.XYZ_to_Lab(color1_xyz)
        color2_lab = self.XYZ_to_Lab(color2_xyz)

        return colour.delta_E(color1_lab, color2_lab, method='CIE 2000')

    def calc_color_difference_Prolab(self, color1_xyz, color2_xyz):
        """ Calculation of the color difference in XYZ color space using Prolab metric """
        color1_prolab = XYZ_to_Prolab(color1_xyz)
        color2_prolab = XYZ_to_Prolab(color2_xyz)

        return np.linalg.norm(color1_prolab - color2_prolab)

    def calc_color_difference(self, color1_xyz, color2_xyz, color_difference_model):
        """ Calculation of the color difference in XYZ color space using one of methods """
        color_difference = {
            'CIEDE2000': self.calc_color_difference_CIEDE2000(color1_xyz, color2_xyz),
            'Prolab': self.calc_color_difference_Prolab(color1_xyz, color2_xyz),
        }
        return color_difference[color_difference_model]

    def get_difference_array(self, array1, array2, norm = 1):
        if (len(array1) == len(array2)):
            difference_array = np.zeros(len(array1))

            for i in range(len(array1)):
                """ chromatic adaptation for observer """
                array1_mod = self.M_CAT.dot(array1[i])

                """ chromatic adaptation for sensor """
                array2_mod = self.M_CAT.dot(array2[i])

                difference_array[i] = self.calc_color_difference(array1_mod, array2_mod, self.color_difference_model)
            return difference_array
        else:
            raise ValueError(
                "An error occurred while calculating the color difference. Arrays have different lengths.")

    def get_difference_array_nonperceptual(self, array1, array2, norm = 2):
        if (len(array1) == len(array2)):
            difference_array = np.zeros(len(array1))

            for i in range(len(array1)):
                difference_array[i] = np.linalg.norm(array1[i] - array2[i], norm)
            return difference_array
        else:
            raise ValueError(
                "An error occurred while calculating the color difference. Arrays have different lengths.")

    def xyz_to_prolab(self, xyz_arr):
        prolab_arr = []
        for val in xyz_arr:
            prolab_arr.append(XYZ_to_Prolab(self.M_CAT.dot(val)))
        return np.array(prolab_arr)

    def get_xyz_and_rgb_input_values(self):
        obj_observer = Color_perseption(self.wl)
        obj_observer.load_CMFs(self.CMFs)
        obj_observer.set_white_illum(self.white_illum)  # here we also found white point using observer CMFs
        #
        obj_sensor = Color_perseption(self.wl)
        obj_sensor.load_SSS(self.SSS)
        obj_sensor.set_white_illum(self.white_illum)
        obj_sensor.set_white_point(obj_observer.white_point)  # here we set white point found for observer CMFs

        N = len(self.reflectances)
        xyz_observer_arr = np.zeros((N, 3))
        rgb_sensor_arr = np.zeros((N, 3))
        for i, color in enumerate(self.reflectances):
            xyz_observer_arr[i] = obj_observer.calculate_xyz_from_spectrum(color)
            rgb_sensor_arr[i] = obj_sensor.calculate_rgb_from_spectrum(color)
        # rgb_sensor_arr *= 255
        self.xyz_observer_arr = xyz_observer_arr
        self.rgb_sensor_arr = rgb_sensor_arr

        return xyz_observer_arr, rgb_sensor_arr

    def reconstruct_reflectances(self, cmf, rgb_array):
        sss_normalized = cmf.T / cmf.mean(axis=1).T
        d, cmfs_w = reflectance_reconstruction_prep(sss_normalized, self.white_illum)

        new_reflectances = []
        for v in rgb_array:
            r = reflectance_reconstruction(d, cmfs_w, v)
            if np.any(r):
                new_reflectances.append(r)

        # coef = sss_normalized.T[1].dot(self.white_illum / 100) / len(self.wl)
        # reflectances = pd.DataFrame(np.array(new_reflectances).T / coef)
        reflectances = pd.DataFrame(np.array(new_reflectances).T)
        reflectances.insert(loc=0, column='wl', value=self.wl)

        return reflectances

    def get_rgb_grid(self, frequency, boundaries):
        r, g, b = np.meshgrid(np.linspace(boundaries[0][0], boundaries[0][1], frequency),
                              np.linspace(boundaries[1][0], boundaries[1][1], frequency),
                              np.linspace(boundaries[2][0], boundaries[2][1], frequency),
                              indexing='ij')
        return np.stack((r, g, b), axis=3).reshape(frequency ** 3, 3)

    def get_data(self):
        """
        :return: raw values for X and Y
        """

        """ chromatic adaptation for observer and sensor"""
        # for i, color in enumerate(self.reflectances):
        #     xyz_observer = self.M_CAT.dot(self.xyz_observer_arr[i])
        #     xyz_sensor = self.M_CAT.dot(self.rgb_sensor_arr)

        return self.rgb_sensor_arr, self.xyz_observer_arr

    """
    Gamut mapping module
    """

class DataGeneratorGamut(object):
    def __init__(self, **kwargs):
        assert all(x in kwargs.keys() for x in ['input_gamut',
                                                'output_gamut',
                                                'boundaries',
                                                'grid_frequency_train',
                                                'grid_frequency_test'])

        self.__dict__.update(kwargs)
        # self.m_cat_to_D65_input_gamut = np.array
        # self.m_cat_to_D65_output_gamut = np.array
        self.m_cat = {'default' : np.array,   # chromatic adaptation matrix for transform from input colorspace to output colorspace
                      'input' : np.array,     # chromatic adaptation matrix for transform from input colorspace to D65
                      'output' : np.array,
                      'input_inv' : np.array,
                      'output_inv' : np.array}

    def rgb_to_xyz(self, val, colorspace):
        rgb = colour.RGB_COLOURSPACES[colorspace]

        return colour.RGB_to_XYZ(
            val,
            rgb.whitepoint,
            rgb.whitepoint,
            rgb.matrix_RGB_to_XYZ,
            'Bradford',
            rgb.cctf_decoding)

    def xyz_to_rgb(self, val, colorspace):
        rgb = colour.RGB_COLOURSPACES[colorspace]

        return colour.XYZ_to_RGB(
            val,
            rgb.whitepoint,
            rgb.whitepoint,
            rgb.matrix_XYZ_to_RGB,
            'Bradford',
            rgb.cctf_encoding)
            # None)

    def xyz_to_rgb_arr(self, xyz_arr, gamut_type = 'output'):
        """
        :param xyz_arr: array of xyz values
        :param gamut_type: input / output
        :return: array of rgb values
        """

        rgb_arr = []
        if gamut_type == 'input':
            gamut = self.input_gamut
        elif gamut_type == 'output':
            gamut = self.output_gamut

        for val in xyz_arr:
            rgb = self.xyz_to_rgb(self.m_cat[gamut_type + '_inv'].dot(val), gamut)

            if np.any(rgb < 0):
                rgb[rgb < 0] = 0
            if np.any(rgb > 1):
                rgb[rgb > 1] = 1

            rgb_arr.append(rgb)
        return np.array(rgb_arr)

    def rgb_to_lab(self, xyz_arr, gamut_type = 'output'):
        """
        :param xyz_arr: array of xyz values
        :param gamut_type: input / output
        :return: array of prolab values
        """

        lab_arr = []
        if gamut_type == 'input':
            gamut = self.input_gamut
        elif gamut_type == 'output':
            gamut = self.output_gamut

        for val in xyz_arr:
            xyz = self.rgb_to_xyz(val, gamut)
            # lab_arr.append(colour.Lab_to_LCHab(colour.XYZ_to_Lab(self.m_cat[gamut_type].dot(xyz))))
            lab_arr.append(colour.XYZ_to_Lab(self.m_cat[gamut_type].dot(xyz)))
        return np.array(lab_arr)

    def rgb_to_prolab(self, xyz_arr, gamut_type = 'output'):
        """
        :param xyz_arr: array of xyz values
        :param gamut_type: input / output
        :return: array of prolab values
        """
        prolab_arr = []
        if gamut_type == 'input':
            gamut = self.input_gamut
        elif gamut_type == 'output':
            gamut = self.output_gamut

        for val in xyz_arr:
            xyz = self.rgb_to_xyz(val, gamut)
            prolab_arr.append(XYZ_to_Prolab(self.m_cat[gamut_type].dot(xyz)))
        return np.array(prolab_arr)

    def calc_color_difference_Prolab(self, color1_xyz, color2_xyz):
        color1_prolab = XYZ_to_Prolab(color1_xyz)
        color2_prolab = XYZ_to_Prolab(color2_xyz)

        return np.linalg.norm(color1_prolab - color2_prolab)

    def get_difference_array(self, array1, array2, norm = 1):
        if (len(array1) == len(array2)):
            difference_array = np.zeros(len(array1))

            for i in range(len(array1)):
                """ chromatic adaptation for observer """
                array1_mod = self.rgb_to_xyz(array1[i], self.output_gamut)
                # array1_mod = self.m_cat_to_D65_output_gamut.dot(array1_mod)
                array1_mod = self.m_cat['output'].dot(array1_mod)

                """ chromatic adaptation for sensor """
                array2_mod = self.rgb_to_xyz(array2[i], self.output_gamut)
                # array2_mod = self.m_cat_to_D65_output_gamut.dot(array2_mod)
                array2_mod = self.m_cat['output'].dot(array2_mod)

                difference_array[i] = self.calc_color_difference_Prolab(array1_mod, array2_mod)
            return difference_array
        else:
            raise ValueError(
                "An error occurred while calculating the color difference. Arrays have different lengths.")

    def in_gamut(self, val):
        # v = self.xyz_to_rgb(val, self.output_gamut)
        v = self.xyz_to_rgb(self.m_cat['output_inv'].dot(colour.Lab_to_XYZ(val)), self.output_gamut)
        # print("v", v)
        return np.all(v > 0) and np.all(v < 1)

    def transform_clipped(self, rgb_input):
        xyz_input = self.rgb_to_xyz(rgb_input, self.input_gamut)
        lab_input = colour.XYZ_to_Lab(self.m_cat['input'].dot(xyz_input))

        # print('v: ', self.xyz_to_rgb(self.m_cat['output_inv'].dot(colour.Lab_to_XYZ(lab_input)), self.output_gamut))
        if not self.in_gamut(lab_input):
            lch_input = colour.Lab_to_LCHab(lab_input)

            # print('rgb input: ', rgb_input)

            def loss_func(x):
                lch = np.array([x[0], x[1], lch_input[2]])
                lab = colour.LCHab_to_Lab(lch)
                if self.in_gamut(lab):
                    return np.linalg.norm(lch_input - lch)
                else:
                    return np.linalg.norm(lch_input - lch) + 1e2

            def loss_grad(x):
                return scipy.optimize.approx_fprime(np.array(x), loss_func, epsilon=1e-6)

            x_init = np.array([lch_input[0], 0])

            # res = scipy.optimize.fmin_bfgs(loss_func, x_init, gtol = 0.99, disp=False)
            # res = scipy.optimize.minimize(loss_func, x_init, method='BFGS', options={'disp': True}).x
            res = scipy.optimize.minimize(loss_func, x_init, method='Nelder-Mead', options={'disp': False}).x

            # print('lch_input', lch_input)
            # print('x_init', x_init)
            # print('res', res)

            lch_output = np.array([res[0], res[1], lch_input[2]])
            lab_output = colour.LCHab_to_Lab(lch_output)
        else:
            lab_output = lab_input

        # print("Lab output: ", lab_output)

        return self.xyz_to_rgb(self.m_cat['output_inv'].dot(colour.Lab_to_XYZ(lab_output)), self.output_gamut)

    def get_data_for_gamut_mapping(self, prerendered = False):
        rgb_input_gamut = colour.RGB_COLOURSPACES[self.input_gamut]
        rgb_output_gamut = colour.RGB_COLOURSPACES[self.output_gamut]

        chromatic_adaptation_transform = "Bradford"
        self.m_cat['default'] = matrix_chromatic_adaptation_VonKries(
            xyY_to_XYZ(xy_to_xyY(rgb_input_gamut.whitepoint)),
            xyY_to_XYZ(xy_to_xyY(rgb_output_gamut.whitepoint)),
            transform=chromatic_adaptation_transform)

        self.m_cat['input'] = matrix_chromatic_adaptation_VonKries(
            xyY_to_XYZ(xy_to_xyY(rgb_input_gamut.whitepoint)),
            [0.9505, 1., 1.089],
            transform=chromatic_adaptation_transform)

        self.m_cat['output'] = matrix_chromatic_adaptation_VonKries(
            xyY_to_XYZ(xy_to_xyY(rgb_output_gamut.whitepoint)),
            [0.9505, 1., 1.089],
            transform=chromatic_adaptation_transform)

        self.m_cat['input_inv'] = np.linalg.inv(self.m_cat['input'])
        self.m_cat['output_inv'] = np.linalg.inv(self.m_cat['output'])

        if prerendered == True:
            rgb_input_train = np.loadtxt('data/prerendered/gamut/prerendered_40_X_train.txt')
            rgb_output_train = np.loadtxt('data/prerendered/gamut/prerendered_40_Y_train.txt')
            rgb_input_test = np.loadtxt('data/prerendered/gamut/prerendered_40_X_train.txt')
            rgb_output_test = np.loadtxt('data/prerendered/gamut/prerendered_40_Y_train.txt')
        else:
            r, g, b = np.meshgrid(np.linspace(self.boundaries[0][0], self.boundaries[0][1], self.grid_frequency_train),
                                  np.linspace(self.boundaries[1][0], self.boundaries[1][1], self.grid_frequency_train),
                                  np.linspace(self.boundaries[2][0], self.boundaries[2][1], self.grid_frequency_train),
                                  indexing='ij')
            rgb_input_train = np.stack((r, g, b), axis=3).reshape(self.grid_frequency_train ** 3, 3)

            r, g, b = np.meshgrid(np.linspace(self.boundaries[0][0], self.boundaries[0][1], self.grid_frequency_test),
                                  np.linspace(self.boundaries[1][0], self.boundaries[1][1], self.grid_frequency_test),
                                  np.linspace(self.boundaries[2][0], self.boundaries[2][1], self.grid_frequency_test),
                                  indexing='ij')
            rgb_input_test = np.stack((r, g, b), axis=3).reshape(self.grid_frequency_test ** 3, 3)

            print("SHAPE: ", rgb_input_train.shape)

            rgb_output_train = []
            for rgb_val in rgb_input_train:
                # rgb_output_train.append(self.xyz_to_rgb(self.m_cat['default'].dot(self.rgb_to_xyz(rgb_val, self.input_gamut)), self.output_gamut))
                rgb_output_train.append(self.transform_clipped(rgb_val))

            rgb_output_test = []
            for rgb_val in rgb_input_test:
                # rgb_output_test.append(self.xyz_to_rgb(self.m_cat['default'].dot(self.rgb_to_xyz(rgb_val, self.input_gamut)), self.output_gamut))
                rgb_output_test.append(self.transform_clipped(rgb_val))

            # np.savetxt("prerendered_40_X_train.txt", rgb_input_train)
            # np.savetxt("prerendered_40_Y_train.txt", rgb_output_train)

        return rgb_input_train, np.array(rgb_output_train), rgb_input_test, np.array(rgb_output_test)
