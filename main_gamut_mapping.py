import numpy as np

from data_generator import DataGeneratorGamut
from interface_class import LUTConstructor
from competitors import LeastSquares
from nonuniform_lut_optiimizer import NonuniformLut
from save_results import *
from competitors import hard_cases_metrics
from hyperspectral_imaging import TestImage
from dataset import Dataset
from pathlib import Path
import time
from competitors import LeastSquares, NotOptimalLUT, hard_cases_metrics
import cv2
import hs_utils

if __name__ == '__main__':
    #------------- INITIALIZE PARAMETERS --------------------------------------------------

    parameters_of_data_generator = {
        'input_gamut': 'P3-D65',  # P3-D65,  ITU-R BT.2020
        'output_gamut': 'sRGB',
        'boundaries': [[0, 1], [0, 1], [0, 1]],
        'grid_frequency_train': 40,
        'grid_frequency_test': 1
    }

    print('\nParameters of data generator')
    for key, value in {**parameters_of_data_generator}.items():
        print(key, ':', value)

    save_results = True
    save_graphs = True

    print("Train data initialization")
    data_generator_obj = DataGeneratorGamut(**parameters_of_data_generator)
    X_train, Y_train, X_test, Y_test = data_generator_obj.get_data_for_gamut_mapping(prerendered=True)

    ######################
    dataset = Dataset()
    """ set train and test reflectances """
    reflectances_train = dataset.reflectances.sfu.macbeth()
    reflectances_test = dataset.reflectances.chromaxion.dc()

    """ set observer sensitivity """
    cmf = dataset.spectral_sensitivity.xyz_matching_fun_31()

    """ set camera sensitivity """
    
    sss = dataset.spectral_sensitivity.sonydxc930()

    white_illuminant = dataset.illuminant.cie.std()
    white_illuminant_name = 'D65' # here always should be D65 !!!

    img_mode = 'hyperspectral'  # 'gradient', 'hyperspectral'
    case = 'GamutInProlab'

    """ hyperspectral imaging """
    hs_image_names = ['2021-07-15_003.h5', '2021-07-17_001.h5', '2021-07-18_003.h5', '2021-07-24_002.h5',
                      '2020-02-25_003.h5', '2020-03-16_015.h5', '2020-03-16_018.h5', '2020-09-15_002.h5',
                      '2020-09-15_005.h5', '2020-09-15_014.h5']
    # hs_image_name = 'none'
    # for i in [4, 5, 6, 7, 8, 9]:
    hs_image_name = 'data/images/' + hs_image_names[7]
    if hs_image_name != 'none':
        test_image_obj = TestImage(case, img_mode)
        test_image_obj.initialize_gamut_generator(data_generator_obj)
        hs_reflectances, hs_illuminant = test_image_obj.initialize(hs_image_name, cmf, sss, white_illuminant,
                                                                   white_illuminant_name)
        _, Y_test = test_image_obj.get_tristimulus_values()
        X_test = np.nan_to_num(data_generator_obj.xyz_to_rgb_arr(Y_test, 'input'))
        # np.savetxt('test_barh_xyz.txt', Y_test)
        # X_test = cv2.normalize(raw_rgb_test, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        X_test_reshaped = X_test.reshape(test_image_obj.get_shape())
        if img_mode == 'gradient':
            reflectances_test[1] = 'gradients'
        else:
            reflectances_test[1] = Path(hs_image_name).stem
    ######################
    # raw_rgb_test = hs_utils.inflateThePicture(raw_rgb_test)
    ######################


    parameters_of_lut = {
        'number_of_nodes_in_R': 9,  # l_r in paper
        'number_of_nodes_in_G': 9,  # l_g in paper
        'number_of_nodes_in_B': 9,  # l_b in paper

        'lambda_R': 0.0001,
        'lambda_S': 0.001,
        'lambda_K': 0.001,
        'calculate_lambda_with_grid_and_train_size': False,  # if any of lambda None there is a two stage procedure for
        # lambda estimation. True here enables the second stage
        'delta_for_lambda': 0.01,  # constant for the first stage of the procedure

        'omega_weights': {
            'R0': 1,
            'R1': 1,
            'G0': 1,
            'G1': 1,
            'B0': 1,
            'B1': 1,
            '000': 0
        },
        'type': 'uniform',  # 'uniform', 'nonuniform'
        'mode': case,  # 'RGB2XYZ', 'RGB2RGB', 'GamutInProlab'
        'interpolation': 'trilinear',  # 'tetrahedral'

        'lambda_contrast': 1,  # 0.000001
        'image_for_contrast': X_test_reshaped,
        'xx_0_for_contrast': 0.01
    }

    #---------------------------------------------------------------------------------------

    print('\nParameters of uniform lut')
    for key, value in {**parameters_of_lut}.items():
        print(key, ':', value)

    """ Least squares processing """
    transform_model = "root_polynomial_3x6"  # linear_3x3, scalable_rational_3x6, scalable_rational_3x10, root_polynomial_3x6
    ls_obj = LeastSquares()
    ls_obj.fit(X_train, Y_train, transform_model)
    Y_train_LS = ls_obj.transform(X_train)
    Y_test_LS = ls_obj.transform(X_test)
    #
    difference_array_train_LS = data_generator_obj.get_difference_array(Y_train, Y_train_LS)
    difference_array_test_LS = data_generator_obj.get_difference_array(Y_test, Y_test_LS)

    print("\nL1 least squares train: ", np.mean(difference_array_train_LS))
    print("L1 least squares test: ", np.mean(difference_array_test_LS))
    #

    """ Not optimal lut processing """
    not_opt_lut_obj = NotOptimalLUT(parameters_of_lut['mode'])
    not_opt_lut_obj.init_GamutInProlab(parameters_of_data_generator)
    not_opt_lut_obj.fit(frequency=9)
    Y_train_NOL = not_opt_lut_obj.transform(X_train)
    Y_test_NOL = not_opt_lut_obj.transform(X_test)

    difference_array_train_NOL = data_generator_obj.get_difference_array(Y_train, Y_train_NOL)
    difference_array_test_NOL = data_generator_obj.get_difference_array(Y_test, Y_test_NOL)

    print("\nL1 not optimal lut train: ", np.mean(difference_array_train_NOL))
    print("L1 not optimal lut test: ", np.mean(difference_array_test_NOL))


    uniform_lut = LUTConstructor(**parameters_of_lut)
    if (parameters_of_lut['type'] == 'nonuniform'):
        nonuniform_lut = NonuniformLut(uniform_lut)
        a_opt, b_opt, image, E_opt, E_color, E_reg, res = nonuniform_lut.train(X_train, Y_train)
        print('a: ', a_opt)
        Y_train_hat = uniform_lut.predict(nonuniform_lut.recalculate_train(res, X_train))
        Y_test_hat = uniform_lut.predict(nonuniform_lut.recalculate_train(res, X_test))

    else:
        a_opt, b_opt, image, E_opt, E_color, E_reg, E_contrast = uniform_lut.train(X_train, Y_train, False)
        Y_train_hat = uniform_lut.predict(X_train)
        Y_test_hat = uniform_lut.predict(X_test)

    results = {
        'E_opt': E_opt,
        'E_color': E_color,
        'E_reg': E_reg,
        'E_reg / E_color': E_reg / E_color,
        'E_contrast': E_contrast
    }

    print('\nResults of optimization')
    for key, value in {**results}.items():
        print(key, ':', value)

        #
    if hs_image_name != 'none':
        print(hs_image_name)
        print('\nLaunch image processing...')
        time_start = time.time()
        data = {
            'OL': Y_test_hat,
            'GT': Y_test,
            'NOL': Y_test_NOL,
            'LS': Y_test_LS,
        }
        test_image_obj.visualize(data, Path(hs_image_name).stem)
        time_stop = time.time()
        show_time(time_start, time_stop, "\nImage processing time: ")
    #

    difference_array_train = data_generator_obj.get_difference_array(Y_train, Y_train_hat)
    difference_array_test = data_generator_obj.get_difference_array(Y_test, Y_test_hat)

    filename = 'mode-GamutInProlab' + 'interpolation-' + parameters_of_lut['interpolation'] + \
               '_input_gamut-' + parameters_of_data_generator['input_gamut'] + \
               '_output_gamut-' + parameters_of_data_generator['output_gamut']

    save_data(filename, {**parameters_of_data_generator, **parameters_of_lut, **results},
              a_opt, b_opt, X_train, Y_train, Y_train_hat, save_results)

    # draw_graph_for_gamut_mapping(filename, parameters_of_data_generator, 'Prolab',
    #                              difference_array_train, difference_array_test, save_results)

    plot_3d_graph(filename, a_opt, b_opt, X_train, Y_train, Y_train_hat, X_test, Y_test, Y_test_hat, save_flag=save_results)

    plot_3d_graph_prolab(filename,
                         data_generator_obj.rgb_to_prolab(X_train, 'input'),
                         data_generator_obj.rgb_to_prolab(Y_train, 'output'),
                         data_generator_obj.rgb_to_prolab(Y_train_hat, 'output'),
                         data_generator_obj.rgb_to_prolab(X_test, 'input'),
                         data_generator_obj.rgb_to_prolab(Y_test, 'output'),
                         data_generator_obj.rgb_to_prolab(Y_test_hat, 'output'),
                         save_flag=True)

    # plot_3d_graph_prolab(filename,
    #                      data_generator_obj.rgb_to_lch_polar(X_train, 'input'),
    #                      data_generator_obj.rgb_to_lch_polar(Y_train, 'output'),
    #                      data_generator_obj.rgb_to_lch_polar(Y_train_hat, 'output'),
    #                      data_generator_obj.rgb_to_lch_polar(X_test, 'input'),
    #                      data_generator_obj.rgb_to_lch_polar(Y_test, 'output'),
    #                      data_generator_obj.rgb_to_lch_polar(Y_test_hat, 'output'),
    #                      save_flag=True)