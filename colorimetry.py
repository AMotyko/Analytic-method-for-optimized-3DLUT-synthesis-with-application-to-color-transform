from dataset import Dataset
from cmf import Color_perseption
from cmf import rgb_to_xyz
import numpy as np
import pandas as pd
import colour
from colour.models import xy_to_xyY, xyY_to_XYZ
from colour.adaptation.vonkries import matrix_chromatic_adaptation_VonKries

class Colorimetry():
    def __init__(self):
        self.wl = np.array
        self.CMF = np.array
        self.white_illum = np.array
        self.M_cat = np.array
        self.prolab_transform_M = np.array([[79.4725, 486.6610, 153.7311, 0],
                               [649.9038, -595.4477, -20.4498, 0],
                               [50.8625, 194.9377, -223.4334, 0],
                               [0.7947, 3.8666, 1.5373, 1]])

    def XYZ_to_Prolab(self, xyz):
        T_h = self.prolab_transform_M.dot(np.append(xyz, 1))
        T_c = np.delete(T_h / T_h[3], 3)
        return T_c

    def initialize_illuminant(self, white_illuminant, illum_name):
        dataset = Dataset()
        cmf = dataset.spectral_sensitivity.xyz_matching_fun_31()[0]
        self.wl = pd.merge(white_illuminant['wl'], cmf['wl'], how='inner').values[:, 0]

        l_cmf = cmf.loc[cmf['wl'].isin(self.wl)]
        l_white_illuminant = white_illuminant.loc[white_illuminant['wl'].isin(self.wl)]

        self.CMFs = l_cmf.drop(columns=['wl']).values.T
        self.white_illum = l_white_illuminant[illum_name].values

        self.obj_observer = Color_perseption(self.wl)
        self.obj_observer.load_CMFs(self.CMFs)
        self.obj_observer.set_white_illum(self.white_illum)
        self.XYZ_ref = self.obj_observer.calculate_xyz_from_spectrum(np.ones(len(self.wl)))

        chromatic_adaptation_transform = "Bradford"
        self.M_CAT_E2Ref = matrix_chromatic_adaptation_VonKries(
            [1, 1, 1],
            self.XYZ_ref,
            transform=chromatic_adaptation_transform)

        self.M_CAT_Ref2D65 = matrix_chromatic_adaptation_VonKries(
            self.XYZ_ref,
            [0.95047, 1, 1.08883],
            transform=chromatic_adaptation_transform)

    def rgb_to_xyz(self, rgb):
        return self.M_CAT_E2Ref.dot(rgb_to_xyz.dot(rgb))

    def calc_color_difference_Prolab(self, color1_xyz, color2_xyz):
        """ Calculation of the color difference in XYZ color space using Prolab metric (chromatic adaptation to D65 included"""

        color1_prolab = self.XYZ_to_Prolab(self.M_CAT_Ref2D65.dot(color1_xyz))
        color2_prolab = self.XYZ_to_Prolab(self.M_CAT_Ref2D65.dot(color2_xyz))

        return np.linalg.norm(color1_prolab - color2_prolab)


"""
Example
"""

dataset = Dataset()

# "E, "D65", "A": dataset.illuminant.cie.std(),
# "FL3.11": dataset.illuminant.cie.fl()
# "LED-RGB100": dataset.illuminant.mls.mls()

""" set white illuminant """
white_illuminant = dataset.illuminant.cie.std()
white_illuminant_name = 'D65'


colorimetry = Colorimetry()
colorimetry.initialize_illuminant(white_illuminant, white_illuminant_name)
