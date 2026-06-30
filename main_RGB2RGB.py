from data_generator import DataGenerator
from dataset import Dataset
from competitors import LeastSquares, NotOptimalLUT
from interface_class import LUTConstructor
from nonuniform_lut_optiimizer import NonuniformLut
from save_results import *
from competitors import hard_cases_metrics

import numpy as np

if __name__ == '__main__':

    dataset = Dataset()

    #------------- INITIALIZE PARAMETERS --------------------------------------------------

    """ set train and test reflectances """
    # reflectances_train = dataset.reflectances.sfu.macbeth()
    reflectances_train = dataset.reflectances.sfu.munsell_extended()
    reflectances_test = dataset.reflectances.babelcolor.sg()

    """ set observer sensitivity """
    cmf = dataset.spectral_sensitivity.xyz_matching_fun_31()

    """ set parameters for reflectances reconstruction"""
    train_grid_frequency = 40
    test_grid_frequency = 16
    boundaries = [[0, 1], [0, 1], [0, 1]]

    """ set camera sensitivity """
    
    sss = dataset.spectral_sensitivity.huawei_p50.sonydxc930()

    """ set white illuminant """
    
    # white_illuminant_name_1 = 'L-LIFX-B'

    # "E": dataset.illuminant.cie.std()
    # "FL3.11": dataset.illuminant.cie.fl()
    # "LED-RGB100": dataset.illuminant.mls.mls()

    white_illuminant_1 = dataset.illuminant.cie.std()
    white_illuminant_name_1 = 'D65'

    white_illuminant_2 = dataset.illuminant.cie.std()
    white_illuminant_name_2 = 'A'

    """ set color difference space ( CIEDE2000, Prolab ) """
    color_difference_model = "L2 in RGB"

    save_results = True
    reconstruct_reflectances_flag = True

    """ create an instance of the class and set the required parameters """
    data_generator_obj = DataGenerator()
    data_generator_obj.set_color_difference_model(color_difference_model)

    print("Train data initialization")
    print("Train data initialization...")
    if reconstruct_reflectances_flag:
        data_generator_obj.set_spectral_data(reflectances_train[0], cmf[0], sss[0], white_illuminant_1,
                                             white_illuminant_name_1,
                                             wl_step=1, print_info=True) # TODO need inner initialzaion for available wavelength
        print("Launch reflectances reconstruction procedure...")
        rgb_grid_train = data_generator_obj.get_rgb_grid(train_grid_frequency, boundaries)
        rgb_grid_test = data_generator_obj.get_rgb_grid(test_grid_frequency, boundaries)
        # reflectances_train = [data_generator_obj.reconstruct_reflectances(data_generator_obj.SSS, rgb_grid_train), 'reconstructed, frequency: '
        #                       + str(train_grid_frequency)]
        reflectances_train = dataset.reflectances.prerendered.grid_40()

        reflectances_test = [data_generator_obj.reconstruct_reflectances(data_generator_obj.SSS, rgb_grid_test), 'reconstructed, frequency: '
                              + str(test_grid_frequency)]

    data_generator_obj.set_spectral_data(reflectances_train[0], cmf[0], sss[0], white_illuminant_1,
                                         white_illuminant_name_1, wl_step=1, print_info=True)

    X_train, _ = data_generator_obj.get_data()

    data_generator_obj.set_spectral_data(reflectances_test[0], cmf[0], sss[0], white_illuminant_1,
                                         white_illuminant_name_1, wl_step=1, print_info=False)
    X_test, _ = data_generator_obj.get_data()

    data_generator_obj.set_spectral_data(reflectances_train[0], cmf[0], sss[0], white_illuminant_2,
                                         white_illuminant_name_2, wl_step=1, print_info=True)

    Y_train, _ = data_generator_obj.get_data()

    data_generator_obj.set_spectral_data(reflectances_test[0], cmf[0], sss[0], white_illuminant_2,
                                         white_illuminant_name_2, wl_step=1, print_info=False)
    Y_test, _ = data_generator_obj.get_data()

########

    parameters_of_lut = {
        'number_of_nodes_in_R': 9,  # l_r in paper
        'number_of_nodes_in_G': 9,  # l_g in paper
        'number_of_nodes_in_B': 9,  # l_b in paper


        'lambda_S': None,
        'lambda_K': None,
        'lambda_R': None,

        'calculate_lambda_with_grid_and_train_size': True,  # if any of lambda None there is a two stage procedure for
        # lambda estimation. True here enables the second stage
        'delta_for_lambda': 0.05,  # constant for the first stage of the procedure

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
        'mode': 'RGB2RGB',  # 'RGB2XYZ', 'RGB2RGB', 'GamutInProlab'
        'interpolation': 'trilinear' #'trilinear'  # 'tetrahedral'
    }

    #---------------------------------------------------------------------------------------

    """ Least squares calculation"""
    transform_model = "root_polynomial_3x6"  # linear_3x3, scalable_rational_3x6, scalable_rational_3x10, root_polynomial_3x6
    ls_obj = LeastSquares()
    ls_obj.fit(X_train, Y_train, transform_model)
    Y_train_LS = ls_obj.transform(X_train)
    Y_test_LS = ls_obj.transform(X_test)

    difference_array_train_LS = data_generator_obj.get_difference_array_nonperceptual(Y_train, Y_train_LS)
    difference_array_test_LS = data_generator_obj.get_difference_array_nonperceptual(Y_test, Y_test_LS)

    print("L1 least squares train: ", np.mean(difference_array_train_LS))
    print("L1 least squares test: ", np.mean(difference_array_test_LS))

    data_LS = Data_for_graphs('LS', 'Least squares', transform_model, difference_array_train_LS,
                              difference_array_test_LS)

    """ Not optimal lut processing """
    not_opt_lut_obj = NotOptimalLUT(parameters_of_lut['mode'])
    not_opt_lut_obj.init_RGB2RGB(cmf, sss, white_illuminant_1, white_illuminant_name_1, white_illuminant_2, white_illuminant_name_2)
    not_opt_lut_obj.fit(frequency=9)
    Y_train_NOL = not_opt_lut_obj.transform(X_train)
    Y_test_NOL = not_opt_lut_obj.transform(X_test)

    difference_array_train_NOL = data_generator_obj.get_difference_array_nonperceptual(Y_train, Y_train_NOL)
    difference_array_test_NOL = data_generator_obj.get_difference_array_nonperceptual(Y_test, Y_test_NOL)

    print("\nL1 not optimal lut train: ", np.mean(difference_array_train_NOL))
    print("L1 not optimal lut test: ", np.mean(difference_array_test_NOL))

    data_NOL = Data_for_graphs('NOL', 'Not optimal lut', 'size: ' + str(not_opt_lut_obj.frequency) + 'x' +
                              str(not_opt_lut_obj.frequency) + 'x' + str(not_opt_lut_obj.frequency),
                               difference_array_train_NOL, difference_array_test_NOL)

    """ Optimal lut processing """
    print('\nParameters of uniform lut')
    for key, value in {**parameters_of_lut}.items():
        print(key, ':', value)

    uniform_lut = LUTConstructor(**parameters_of_lut)
    if (parameters_of_lut['type'] == 'nonuniform'):
        nonuniform_lut = NonuniformLut(uniform_lut)
        a_opt, b_opt, image, E_opt, E_color, E_reg, res = nonuniform_lut.train(X_train, Y_train)
        print('a: ', a_opt)
        Y_train_hat = uniform_lut.predict(nonuniform_lut.recalculate_train(res, X_train))
        Y_test_hat = uniform_lut.predict(nonuniform_lut.recalculate_train(res, X_test))

    else:
        a_opt, b_opt, image, E_opt, E_color, E_reg = uniform_lut.train(X_train, Y_train, False)
        Y_train_hat = uniform_lut.predict(X_train)
        Y_test_hat = uniform_lut.predict(X_test)

    results = {
        'E_opt': E_opt,
        'E_color': E_color,
        'E_reg': E_reg,
        'E_reg / E_color': E_reg / E_color
    }

    print('\nResults of optimization')
    for key, value in {**results}.items():
        print(key, ':', value)

    difference_array_train_OL = data_generator_obj.get_difference_array_nonperceptual(Y_train, Y_train_hat)
    difference_array_test_OL = data_generator_obj.get_difference_array_nonperceptual(Y_test, Y_test_hat)

    data_OL = Data_for_graphs('OL', 'Optimal LUT', 'size: ' + str(parameters_of_lut['number_of_nodes_in_R']) + 'x' +
                              str(parameters_of_lut['number_of_nodes_in_G']) + 'x' +
                              str(parameters_of_lut['number_of_nodes_in_B']),
                              difference_array_train_OL, difference_array_test_OL)

    filename = 'mode-RGB2RGB' + '_interpolation-' + parameters_of_lut['interpolation'] + '_sss-' + sss[1] + \
               '_illum1-' + white_illuminant_name_1 + 'illum2-' + white_illuminant_name_2

    save_data(filename, {**parameters_of_lut, **results}, a_opt, b_opt, X_train, Y_train, Y_train_hat, save_results)

    data_for_graphs = [data_OL, data_LS, data_NOL]

    draw_graph('type-' + parameters_of_lut['type'] +
               '_interpolation-' + parameters_of_lut['interpolation'],
               sss[1], reflectances_train[1], reflectances_test[1], [white_illuminant_name_1, white_illuminant_name_2],
               color_difference_model, data_for_graphs, save_results)

    plot_3d_graph(filename, a_opt, b_opt, X_train, Y_train, Y_train_hat, X_test, Y_test, Y_test_hat, save_flag=save_results)
