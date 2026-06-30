from data_generator import DataGenerator
from save_results import draw_graph, Data_for_graphs, save_data, plot_3d_graph, show_time
from hyperspectral_imaging import TestImage
from dataset import Dataset
from interface_class import LUTConstructor
from nonuniform_lut_optiimizer import NonuniformLut
from competitors import LeastSquares, NotOptimalLUT, hard_cases_metrics
import time
import numpy as np
import pandas as pd
from pathlib import Path


if __name__ == '__main__':

    dataset = Dataset()

    #------------- INITIALIZE PARAMETERS --------------------------------------------------

    """ set train and test reflectances """
    reflectances_train = dataset.reflectances.sfu.macbeth()
    # reflectances_train = dataset.reflectances.chromaxion.dc()
    # reflectances_train = dataset.reflectances.babelcolor.sg()
    # reflectances_train = dataset.reflectances.sfu.munsell_extended()
    reflectances_test = dataset.reflectances.chromaxion.dc()

    """ set observer sensitivity """
    cmf = dataset.spectral_sensitivity.xyz_matching_fun_31()

    """ set parameters for reflectances reconstruction"""
    train_grid_frequency = 10
    test_grid_frequency = 15
    boundaries = [[0, 1], [0, 1], [0, 1]]

    """ set camera sensitivity """
    
    
    sss = dataset.spectral_sensitivity.sonydxc930()

    # "E": dataset.illuminant.cie.std()
    # "FL3.11": dataset.illuminant.cie.fl()
    # "LED-RGB100": dataset.illuminant.mls.mls()

    """ set white illuminant """
   
    # white_illuminant_name = 'A18'

    white_illuminant = dataset.illuminant.cie.std()
    white_illuminant_name = 'D65'

    """ set color difference space ( CIEDE2000, Prolab ) """
    color_difference_model = "Prolab"

    img_mode = 'hyperspectral'  # 'gradients', 'hyperspectral'
    case = 'RGB2XYZ'

    save_results_flag = True
    reconstruct_reflectances_flag = True

    """ create an instance of the class and set the required parameters """
    data_generator_obj = DataGenerator()
    data_generator_obj.set_color_difference_model(color_difference_model)

    print("Train data initialization...")
    if reconstruct_reflectances_flag:
        data_generator_obj.set_spectral_data(reflectances_train[0], cmf[0], sss[0], white_illuminant,
                                             white_illuminant_name,
                                             wl_step=1, print_info=True) # TODO need inner initialzaion for available wavelength
        print("Launch reflectances reconstruction procedure...")
        rgb_grid_train = data_generator_obj.get_rgb_grid(train_grid_frequency, boundaries)
        # rgb_grid_test = data_generator_obj.get_rgb_grid(test_grid_frequency, boundaries)
        # reflectances_train = [data_generator_obj.reconstruct_reflectances(data_generator_obj.SSS, rgb_grid_train), 'reconstructed, frequency: '
        #                       + str(train_grid_frequency)]
        reflectances_train = dataset.reflectances.prerendered.grid_40_p50_D65()

    data_generator_obj.set_spectral_data(reflectances_train[0], cmf[0], sss[0], white_illuminant, white_illuminant_name,
                                           wl_step=1, print_info=True)
    X_train, Y_train = data_generator_obj.get_data()

    data_generator_obj.set_spectral_data(reflectances_test[0], cmf[0], sss[0], white_illuminant,
                                         white_illuminant_name,
                                         wl_step=1, print_info=False)
    X_test, Y_test = data_generator_obj.get_data()

    """ hyperspectral imaging """
    hs_image_names = ['2021-07-15_003.h5', '2021-07-17_001.h5', '2021-07-18_003.h5', '2021-07-24_002.h5',
                      '2020-02-25_003.h5', '2020-03-16_015.h5', '2020-03-16_018.h5', '2020-09-15_002.h5',
                      '2020-09-15_005.h5', '2020-09-15_014.h5']
    # hs_image_name = 'none'
    # for i in [4, 5, 6, 7, 8, 9]:
    hs_image_name = 'data/images/' + hs_image_names[7]
    if hs_image_name != 'none':
        test_image_obj = TestImage(case, img_mode)
        hs_reflectances, hs_illuminant = test_image_obj.initialize(hs_image_name, cmf, sss, white_illuminant,
                                                                   white_illuminant_name)
        X_test, Y_test = test_image_obj.get_tristimulus_values()
        if img_mode == 'gradients':
            reflectances_test[1] = 'gradients'
            X_test_reshaped = None
        else:
            reflectances_test[1] = Path(hs_image_name).stem
            X_test_reshaped = X_test.reshape(test_image_obj.get_shape())

    """ Optimal lut """
    parameters_of_lut = {
        'number_of_nodes_in_R': 9,  # l_r in paper
        'number_of_nodes_in_G': 9,  # l_g in paper
        'number_of_nodes_in_B': 9,  # l_b in paper

        'lambda_R': 0.00001,
        'lambda_S': 0.0001,
        'lambda_K': 0.0001,
        'calculate_lambda_with_grid_and_train_size': True,  # if any of lambda None there is a two stage procedure for
        # lambda estimation. True here enables the second stage
        'delta_for_lambda': 0.1,  # constant for the first stage of the procedure

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
        'mode': 'RGB2XYZ',  # 'RGB2XYZ', 'RGB2RGB', 'GamutInProlab'
        'interpolation': 'trilinear',  # 'tetrahedral'
        'weighted': True,

        'lambda_contrast': 0.01,
        'image_for_contrast': X_test_reshaped,
        'xx_0_for_contrast': 0.01,
        'gram': np.identity(3),
        'gram_y': np.identity(3)
    }

    # #---------------------------------------------------------------------------------------

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

    data_LS = Data_for_graphs('LS', 'Least squares', transform_model, difference_array_train_LS, difference_array_test_LS)
    #

    """ Not optimal lut processing """
    not_opt_lut_obj = NotOptimalLUT(parameters_of_lut['mode'])
    not_opt_lut_obj.init_RGB2XYZ(cmf, sss, white_illuminant, white_illuminant_name)
    not_opt_lut_obj.fit(frequency=9)
    Y_train_NOL = not_opt_lut_obj.transform(X_train)
    Y_test_NOL = not_opt_lut_obj.transform(X_test)

    difference_array_train_NOL = data_generator_obj.get_difference_array(Y_train, Y_train_NOL)
    difference_array_test_NOL = data_generator_obj.get_difference_array(Y_test, Y_test_NOL)

    print("\nL1 not optimal lut train: ", np.mean(difference_array_train_NOL))
    print("L1 not optimal lut test: ", np.mean(difference_array_test_NOL))

    data_NOL = Data_for_graphs('NOL', 'Not optimal lut', 'size_' + str(not_opt_lut_obj.frequency) + 'x' +
                              str(not_opt_lut_obj.frequency) + 'x' + str(not_opt_lut_obj.frequency),
                               difference_array_train_NOL, difference_array_test_NOL)

    """ Optimal lut processing """
    print('\nParameters of uniform lut:')
    for key, value in {**parameters_of_lut}.items():
        print(key, ':', value)

    time_start = time.time()
    uniform_lut = LUTConstructor(**parameters_of_lut)
    if (parameters_of_lut['type'] == 'nonuniform'):
        nonuniform_lut = NonuniformLut(uniform_lut)
        if parameters_of_lut['lambda_contrast'] is None:
            a_opt, b_opt, image, E_opt, E_color, E_reg, res = nonuniform_lut.train(X_train, Y_train)
            E_contrast = None
        else:
            a_opt, b_opt, image, E_opt, E_color, E_reg, res, E_contrast = nonuniform_lut.train(X_train, Y_train)
        print('a: ', a_opt)
        Y_train_hat = uniform_lut.predict(nonuniform_lut.recalculate_train(res, X_train))
        Y_test_hat = uniform_lut.predict(nonuniform_lut.recalculate_train(res, X_test))

    else:
        if parameters_of_lut['lambda_contrast'] is None:
            a_opt, b_opt, image, E_opt, E_color, E_reg = uniform_lut.train(X_train, Y_train, False)
            E_contrast = None
        else:
            a_opt, b_opt, image, E_opt, E_color, E_reg, E_contrast = uniform_lut.train(X_train, Y_train, False)
        Y_train_hat = uniform_lut.predict(X_train)
        Y_test_hat = uniform_lut.predict(X_test)

    time_stop = time.time()
    show_time(time_start, time_stop, "LUT created successfully. Total elapsed time: ")

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
            'GT' : Y_test,
            'NOL' : Y_test_NOL,
            'LS' : Y_test_LS,
        }
        test_image_obj.visualize(data, Path(hs_image_name).stem)
        time_stop = time.time()
        show_time(time_start, time_stop, "\nImage processing time: ")
    #
    difference_array_train_OL = data_generator_obj.get_difference_array(Y_train, Y_train_hat)
    difference_array_test_OL = data_generator_obj.get_difference_array(Y_test, Y_test_hat)

    print("\nL1 optimal lut train: ", np.mean(difference_array_train_OL))
    print("L1 optimal lut test: ", np.mean(difference_array_test_OL))

    data_OL = Data_for_graphs('OL', 'Optimal LUT', 'size_' + str(parameters_of_lut['number_of_nodes_in_R']) + 'x' +
                              str(parameters_of_lut['number_of_nodes_in_G']) + 'x' +
                              str(parameters_of_lut['number_of_nodes_in_B']),
                              difference_array_train_OL, difference_array_test_OL)

    #
    hc = hard_cases_metrics(difference_array_train_OL, difference_array_test_OL)
    print('Vector for hardcases', hc)
    #
    filename = Path(hs_image_name).stem + '_mode-RGB2XYZ' + '_interpolation-' + parameters_of_lut['interpolation'] + '_sss-' + sss[1] + \
               '_illum-' + white_illuminant_name
    #
    # # save_data(filename, {**parameters_of_lut, **results}, a_opt, b_opt, X_train, Y_train, Y_train_hat, save_results)
    # #

    data_for_graphs = [data_OL, data_LS, data_NOL]


    draw_graph('type-' + parameters_of_lut['type'] +
               '_interpolation-' + parameters_of_lut['interpolation'],
               sss[1], reflectances_train[1], reflectances_test[1], white_illuminant_name,
               color_difference_model, data_for_graphs, save_results_flag)
    #
    plot_3d_graph(filename, a_opt, b_opt, X_train, Y_train, Y_train_hat, X_test, Y_test, Y_test_hat, save_flag=save_results_flag)
    # # plot_3d_graph_prolab(filename,
    # #                      data_generator_obj.xyz_to_prolab(Y_train),
    # #                      data_generator_obj.xyz_to_prolab(Y_train_hat),
    # #                      data_generator_obj.xyz_to_prolab(Y_test),
    # #                      data_generator_obj.xyz_to_prolab(Y_test_hat),
    # #                      save_flag=True)