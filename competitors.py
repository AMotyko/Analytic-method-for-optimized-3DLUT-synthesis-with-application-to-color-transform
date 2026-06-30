from not_optimal_lut_construction import IrregularSearchSpace, TrilinearInterpolation, NotOptimalLUTGrid, \
    UniformSearchSpace
from dataset import Dataset
from data_generator import DataGenerator, DataGeneratorGamut
import numpy as np
import copy

def hard_cases_metrics(difference_array_CIEDE_L1_LMS, difference_array_CIEDE_L1_Global):
    hc_after_lms_1 = sum(x <= 1 for x in difference_array_CIEDE_L1_LMS.ravel())
    hc_after_lms_3 = sum(x <= 3 for x in difference_array_CIEDE_L1_LMS.ravel())
    hc_after_lms_5 = sum(x <= 5 for x in difference_array_CIEDE_L1_LMS.ravel())
    hc_after_lms_max = sum(x > 5 for x in difference_array_CIEDE_L1_LMS.ravel())

    hc_after_lms_max
    hc_after_lms_5 = hc_after_lms_5 - hc_after_lms_3
    hc_after_lms_3 = hc_after_lms_3 - hc_after_lms_1

    hc_after_glb_1 = sum(x <= 1 for x in difference_array_CIEDE_L1_Global.ravel())
    hc_after_glb_3 = sum(x <= 3 for x in difference_array_CIEDE_L1_Global.ravel())
    hc_after_glb_5 = sum(x <= 5 for x in difference_array_CIEDE_L1_Global.ravel())
    hc_after_glb_max = sum(x > 5 for x in difference_array_CIEDE_L1_Global.ravel())

    hc_after_glb_max
    hc_after_glb_5 = hc_after_glb_5 - hc_after_glb_3
    hc_after_glb_3 = hc_after_glb_3 - hc_after_glb_1

    lms_score = hc_after_lms_3 + hc_after_lms_5 * 4 + hc_after_lms_max * 9
    glb_score = hc_after_glb_3 + hc_after_glb_5 * 4 + hc_after_glb_max * 9

    return [hc_after_lms_1, hc_after_lms_3, hc_after_lms_5, hc_after_lms_max, lms_score, hc_after_glb_1, hc_after_glb_3, hc_after_glb_5, hc_after_glb_max, glb_score]

class LeastSquares(object):
    def __init__(self):
        self.available_parameters = {
            'model': ['linear_3x3', 'scalable_rational_3x6', 'scalable_rational_3x10', 'root_polynomial_3x6']
        }
        self.transform_model = str
        self.coef = np.array

    def fit(self, S, O, model='linear_3x3'):
        if model in self.available_parameters['model']:
            self.transform_model = model
            r = S[:, 0]
            g = S[:, 1]
            b = S[:, 2]
            param = np.array
            if model == "linear_3x3":
                param = [r, g, b]
            elif model == "scalable_rational_3x6":
                sum = r + g + b
                param = [r, g, b, r * g / sum, r * b / sum, g * b / sum]
            elif model == "scalable_rational_3x10":
                sum = r + g + b
                param = [r, g, b, r * g / sum, r * b / sum, g * b / sum,
                         r * g * g / sum / sum, g * b * b / sum / sum, r * b * b / sum / sum,
                         r * g * b / sum / sum]
            elif model == "root_polynomial_3x6":
                param = [r, g, b, (r * g) ** (1 / 2.), (r * b) ** (1 / 2.), (g * b) ** (1 / 2.)]

            S_mod = np.column_stack(param)
            self.coef = np.transpose(np.linalg.inv(S_mod.T.dot(S_mod)).dot(S_mod.T).dot(O))
        else:
            raise ValueError(
                "Parameter '" + model + "' for competitor is incorrect. Please, use one of available parameters: "
                + str(self.available_parameters['model']))


    def transform(self, input_rgb):
        def transform_color(input_rgb_color):
            if self.transform_model == 'linear_3x3':
                return self.coef.dot(input_rgb_color)
            else:
                r = input_rgb_color[0]
                g = input_rgb_color[1]
                b = input_rgb_color[2]
                sum = r + g + b

                if sum == 0:
                    return np.array([0, 0, 0])
                else:
                    param = np.array
                    if self.transform_model == "scalable_rational_3x10":
                        param = [r, g, b, r * g / sum, r * b / sum, g * b / sum,
                                 r * g * g / sum / sum, g * b * b / sum / sum, r * b * b / sum / sum,
                                 r * g * b / sum / sum]
                    elif self.transform_model == "scalable_rational_3x6":
                        param = [r, g, b, r * g / sum, r * b / sum, g * b / sum]
                    elif self.transform_model == "root_polynomial_3x6":
                        param = [r, g, b, (r * g) ** (1 / 2.), (r * b) ** (1 / 2.), (g * b) ** (1 / 2.)]

                    return self.coef.dot(np.array(param))

        transformed_colors = []
        for color in input_rgb:
            transformed_colors.append(transform_color(color))

        return np.array(transformed_colors)

class NotOptimalLUT():
    def __init__(self, case):
        self.case = case
        self.frequency = 9
        self.lut = NotOptimalLUTGrid

    def init_RGB2XYZ(self, cmf, sss, white_illum, white_illum_name):
        self.cmf = cmf
        self.sss = sss
        self.white_illuminant = white_illum
        self.white_illuminant_name = white_illum_name

    def init_RGB2RGB(self, cmf, sss, white_illuminant_1, white_illuminant_name_1, white_illuminant_2, white_illuminant_name_2):
        self.cmf = cmf
        self.sss = sss
        self.white_illuminant_1 = white_illuminant_1
        self.white_illuminant_name_1 = white_illuminant_name_1
        self.white_illuminant_2 = white_illuminant_2
        self.white_illuminant_name_2 = white_illuminant_name_2

    def init_GamutInProlab(self, param):
        self.parameters_of_data_generator = param

    def fit(self, frequency):
        self.frequency = frequency
        parameters_of_lut = {
            'not_opt_grid_frequency': frequency,
            'type': 'not_opt_grid',
            'interpolation': 'trilinear'
        }
        not_opt_grid_frequency = parameters_of_lut['not_opt_grid_frequency']

        # rgb_grid_train = data_generator_obj.get_rgb_grid(not_opt_grid_frequency, boundaries)
        aldas = UniformSearchSpace(not_opt_grid_frequency, not_opt_grid_frequency, not_opt_grid_frequency)
        vertices_index = aldas.all_vertices_single.keys()
        vertices_points = aldas.all_vertices_single.values()
        vertices_index, vertices_points = zip(*sorted(zip(vertices_index, vertices_points)))
        vertices_index = list(vertices_index)
        vertices_points = np.array(list(vertices_points))
        rgb_grid_train = vertices_points

        print("\nReconstruction of reflectances for not optimal lut...")
        if self.case == 'RGB2XYZ':
            data_generator_obj = DataGenerator()
            dataset = Dataset()
            reflectances_train = dataset.reflectances.sfu.macbeth()
            data_generator_obj.set_spectral_data(reflectances_train[0], self.cmf[0],
                                                 self.sss[0], self.white_illuminant,
                                                 self.white_illuminant_name,
                                                 wl_step=1,
                                                 print_info=False)

            reflectances_train = [
                data_generator_obj.reconstruct_reflectances(data_generator_obj.SSS, rgb_grid_train),
                'reconstructed, frequency: ' + str(not_opt_grid_frequency)]

            data_generator_obj.set_spectral_data(reflectances_train[0], self.cmf[0],
                                                             self.sss[0], self.white_illuminant,
                                                             self.white_illuminant_name,
                                                             wl_step=1, print_info=False)
            X_train, Y_train = data_generator_obj.get_data()

        elif self.case == 'RGB2RGB':
            data_generator_obj = DataGenerator()
            dataset = Dataset()
            reflectances_train = dataset.reflectances.sfu.macbeth()
            data_generator_obj.set_spectral_data(reflectances_train[0], self.cmf[0],
                                                 self.sss[0], self.white_illuminant_1,
                                                 self.white_illuminant_name_1,
                                                 wl_step=1,
                                                 print_info=False)
            reflectances_train = [
                data_generator_obj.reconstruct_reflectances(data_generator_obj.SSS, rgb_grid_train),
                'reconstructed, frequency: ' + str(not_opt_grid_frequency)]

            data_generator_obj.set_spectral_data(reflectances_train[0], self.cmf[0],
                                                 self.sss[0], self.white_illuminant_1,
                                                 self.white_illuminant_name_1,
                                                 wl_step=1,
                                                 print_info=False)

            X_train, _ = data_generator_obj.get_data()

            data_generator_obj.set_spectral_data(reflectances_train[0], self.cmf[0],
                                                 self.sss[0], self.white_illuminant_2,
                                                 self.white_illuminant_name_2,
                                                 wl_step=1,
                                                 print_info=False)

            Y_train, _ = data_generator_obj.get_data()

        elif self.case == 'GamutInProlab':
            new_params = copy.deepcopy(self.parameters_of_data_generator)
            new_params['grid_frequency_train'] = frequency
            new_params['grid_frequency_test'] = 1
            data_generator_obj = DataGeneratorGamut(**new_params)
            X_train, Y_train, X_test, Y_test = data_generator_obj.get_data_for_gamut_mapping()

        space = IrregularSearchSpace(rgb_grid_train, X_train, Y_train, aldas.single_to_multi_map)
        trilinear_int = TrilinearInterpolation(space)
        self.lut = NotOptimalLUTGrid(trilinear_int, space)
        # a_opt = rgb_grid_train

    def transform(self, input_rgb):
        return self.lut.predict(input_rgb)