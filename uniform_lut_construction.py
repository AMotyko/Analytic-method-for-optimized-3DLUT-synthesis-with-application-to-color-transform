import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from uniform_search_space import UniformSearchSpace, TrilinearInterpolation, Interpolation, plot_dict, timing, TetrahedralInterpolation
import sys
from typing import Callable, Tuple, Union, Optional
from colorimetry import Colorimetry, Dataset
from jit_functions import numba_get_optimal_b
import scipy


def get_proper_prolab_function(white_illuminant_name):
    if 'GT' in white_illuminant_name:
        white_illuminant_name = 'E'
    elif 'RGB' in white_illuminant_name:
        white_illuminant_name = 'E'
    dataset = Dataset()
    white_illuminant = dataset.illuminant.cie.std()

    colorimetry = Colorimetry()
    colorimetry.initialize_illuminant(white_illuminant, white_illuminant_name)
    return colorimetry.calc_color_difference_Prolab


def check_symmetric(a, rtol=1e-08, atol=1e-08) -> bool:
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def get_diagonal_matrix_with_vals(*args) -> np.array:
    shape = (len(args), len(args))
    mat = np.zeros(shape)
    for index, value in enumerate(args):
        mat[index, index] = value
    return mat


def h_contrast(x: float) -> float:
    """
    function for sigma
    :param x: coordinate
    :return: result
    """
    return min(1, max(0, 20 * (0.1 - x), 20 * (x - 0.9)))


def sigma_contrast(x_tilda: Union[np.array, Tuple]) -> float:
    """

    :param x:
    :return:
    """

    return max([h_contrast(x) for x in x_tilda])


class UniformGridSearchFromMatrices(object):
    def __init__(self, space: UniformSearchSpace, interpolation: Interpolation,
                 X: np.array, Y: np.array,
                 lambda_R: Union[float, None],
                 lambda_S: Union[float, None],
                 lambda_K: Union[float, None],
                 omega_weights: dict, delta_for_lambda: float,
                 calculate_lambda_with_grid_and_train_size: bool = False,
                 weighted: bool = False,
                 plot_graph: bool = True,
                 lambda_contrast: Union[float, None] = 0,
                 image_for_contrast: Union[np.array, None] = None,
                 xx_0_for_contrast: float = 0.01,  # SAW to choose xx_0 between 0.001 and 0.01
                 gram: np.array = np.identity(3),
                 gram_y: np.array = np.identity(3)
                 ) -> None:

        self.plot_graph = plot_graph
        self.space = space
        self.l_r = self.space.l_r  #
        self.l_g = self.space.l_g  # fixme inherit both interpolation and space
        self.l_b = self.space.l_b  #
        self.m = (self.l_r + 1) * (self.l_g + 1) * (self.l_b + 1)
        self.X = X
        self.Y = Y
        self.interpolation = interpolation
        self.omega_weights = omega_weights

        self.q_bold = np.array([[0.7554, 3.8666, 1.6739]])
        self.Q = np.array([[0.755362, 4.86661, 1.67387],
                           [6.17714, -5.95448, -0.222664],
                           [0.483433, 1.94938, -2.43281]])
        self.V = list()
        self.F = np.zeros((3 * self.m, 3 * self.m))
        self.g_bold = np.zeros((3 * self.m))
        self.S = np.zeros((3 * self.m, 3 * self.m))
        self.K = np.zeros((3 * self.m, 3 * self.m))
        self.R = np.zeros((3 * self.m, 3 * self.m))
        self.s_bold = np.zeros((3 * self.m))
        self.h = 0  # TODO check
        self.t = 0  # TODO check
        self.full_dict_with_weights_for_all_X = {}
        self.b_opt = np.zeros((3 * self.m))
        self.weighted = weighted
        self.trained = False
        self.T_contrast = np.zeros((3 * self.m, 3 * self.m))
        self.u_contrast = np.zeros((3 * self.m))
        self.v_contrast = 0
        self.xx_0 = xx_0_for_contrast
        self.image_X = image_for_contrast
        self.Gram = gram
        self.Gram_y = gram_y
        assert self.Gram.shape == (3, 3), f"Wrong Gram matrix: shape is {self.Gram.shape}, should be (3, 3)"
        assert self.Gram_y.shape == (3, 3), f"Wrong Gram matrix: shape is {self.Gram_y.shape}, should be (3, 3)"

        self.lambda_contrast = lambda_contrast if lambda_contrast else 0
        if self.image_X is None:
            self.lambda_contrast = 0

        self.calculate()  # FIXME

        self.delta = delta_for_lambda
        if None in [lambda_S, lambda_K, lambda_R]:
            lambda_S_precalc, lambda_K_precalc, lambda_R_precalc = \
                self.get_lambdas_from_grid_size(calculate_lambda_with_grid_and_train_size)
            self.lambda_S = lambda_S if lambda_S else lambda_S_precalc
            self.lambda_K = lambda_K if lambda_K else lambda_K_precalc
            self.lambda_R = lambda_R if lambda_R else lambda_R_precalc
        else:
            self.lambda_S = lambda_S
            self.lambda_K = lambda_K
            self.lambda_R = lambda_R

        if self.lambda_contrast:
            self.get_T_u_v_for_contrast()

    def validate_results(self):
        E_color = float(self.get_E_color(self.b_opt))
        E_reg = float(self.get_E_reg(self.lambda_S, self.lambda_K, self.lambda_R, self.b_opt))
        if E_color == 0:
            val_num = np.Inf
        else:
            val_num = E_reg / E_color
        to_print = ''
        if 0.1 <= val_num <= 0.6:
            to_print = 'Reasonable result'
        elif val_num < 0.1:
            to_print = 'the solution found underestimates the regularization'
        elif val_num > 0.6:
            to_print = 'the solution found overestimates the regularization'
        logging.debug(f'E_color = {E_color}')
        logging.debug(f'E_reg = {E_reg}')
        logging.debug(f'E_reg / E_color = {val_num}, {to_print}')

    def update_lambdas_procedure(self, b_opt: np.array, with_LUT_and_train_size: bool):

        lambda_tilda_S = self.delta * self.get_E_color(b_opt) / self.get_square_form_S(b_opt)
        lambda_tilda_K = self.delta * self.get_E_color(b_opt) / self.get_square_form_K(b_opt)
        lambda_tilda_R = self.delta * self.get_E_color(b_opt) / self.get_square_form_ro(b_opt)

        logging.debug(f'lambda  S tilda = {lambda_tilda_S}')
        logging.debug(f'lambda  K tilda = {lambda_tilda_K}')
        logging.debug(f'lambda  R tilda = {lambda_tilda_R}')

        if not with_LUT_and_train_size:
            return lambda_tilda_S, lambda_tilda_K, lambda_tilda_R
        else:

            assert self.l_r == self.l_g == self.l_b, "This Lambda estimation procedure can only be used for same number of nodes"

            final_lambda_S = (((self.l_r + 1) ** 2) * (self.l_r - 1)) * lambda_tilda_S / len(self.Y)
            final_lambda_K = self.l_r * lambda_tilda_K / len(self.Y)
            final_lambda_R = ((self.l_r + 1) ** 2) * lambda_tilda_R / len(self.Y)

            logging.debug(f'final_lambda_S = {final_lambda_S}')
            logging.debug(f'final_lambda_K = {final_lambda_K}')
            logging.debug(f'final_lambda_R = {final_lambda_R}')

            return final_lambda_S, final_lambda_K, final_lambda_R

    def get_lambdas_from_grid_size(self, with_LUT_and_train_size: bool = False) -> Tuple[float, float, float]:

        lambda_0_S = self.delta * np.linalg.norm(self.F, ord=1) / np.linalg.norm(self.S, ord=1)
        lambda_0_K = self.delta * np.linalg.norm(self.F, ord=1) / np.linalg.norm(self.K, ord=1)
        lambda_0_R = self.delta * np.linalg.norm(self.F, ord=1) / np.linalg.norm(self.R, ord=1)
        self.b_opt = self.get_optimal_b(lambda_0_S, lambda_0_K, lambda_0_R, self.lambda_contrast)
        for i in range(10000):

            prev_lambda_0_S, prev_lambda_0_K, prev_lambda_0_R = lambda_0_S, lambda_0_K, lambda_0_R
            lambda_0_S, lambda_0_K, lambda_0_R = self.update_lambdas_procedure(self.b_opt, with_LUT_and_train_size)

            self.b_opt = self.get_optimal_b(lambda_0_S, lambda_0_K, lambda_0_R, self.lambda_contrast)
            E_color = float(self.get_E_color(self.b_opt))
            E_reg = float(self.get_E_reg(lambda_0_S, lambda_0_K, lambda_0_R, self.b_opt))

            print(f'Current E_reg / E_color = {E_reg / E_color}')
            print(f'Current lambda_S, lambda_K, lambda_R = {lambda_0_S, lambda_0_K, lambda_0_R}')

            if 0.1 < E_reg / E_color < 0.6:
                print('Optimum reached')
                return lambda_0_S, lambda_0_K, lambda_0_R

            if np.allclose([lambda_0_S, lambda_0_K, lambda_0_R], [prev_lambda_0_S, prev_lambda_0_K, prev_lambda_0_R],
                           rtol=1e-08, atol=1e-08):
                print('Converged')
                return lambda_0_S, lambda_0_K, lambda_0_R

        return lambda_0_S, lambda_0_K, lambda_0_R

    def get_square_form_ro(self, b_opt: np.array) -> float:
        return float(b_opt.T.dot(self.R).dot(b_opt) + 2 * self.s_bold.T.dot(b_opt) + self.t)

    def get_square_form_S(self, b_opt: np.array) -> float:
        return b_opt.T.dot(self.S).dot(b_opt)

    def get_square_form_K(self, b_opt: np.array) -> float:
        return b_opt.T.dot(self.K).dot(b_opt)

    def get_V(self, filename=''):
        if filename:
            weights_v = np.genfromtxt(filename, delimiter=',')  # TODO fix this; only after examples of weights
        else:
            weights_v = np.ones((len(self.X), 3))  
            # TODO is it per color (number of training points) or per channel(3)
            self.V = list()
            for i in range(len(self.Y)):
                reshaped_Y_i = self.Y[i].reshape((1, 3))  # TODO assert same results
                self.V.append((1 / (1 + self.q_bold.dot(self.Y[i]))) * np.array([[weights_v[i][0], 0, 0],
                                                                                 [0, weights_v[i][1], 0],
                                                                                 [0, 0, weights_v[i][2]]]))

    @timing
    def get_F_and_g_bold_and_h(self) -> None:
        self.get_V()

        # ~~~~~~~~~ GET F and g_bold
        self.F = np.zeros((3 * self.m, 3 * self.m))
        self.g_bold = np.zeros((3 * self.m))

        for index, cur_train_point in enumerate(self.X):
            _, weights_for_grid_si = self.interpolation.get_weights_for_full_grid(cur_train_point)
            full_dict_with_QVQ_for_all_X = self.Q.T.dot(self.V[index]).dot(self.Q)
            full_dict_with_YQVQ_for_all_X = self.Y[index].dot(self.Q.T).dot(self.V[index]).dot(self.Q)

            for col in range(3):
                for j, weight in weights_for_grid_si.items():
                    self.g_bold[col * self.m + j] += weight * full_dict_with_YQVQ_for_all_X[col]
                    for col_ in range(3):
                        for j_, weight_ in weights_for_grid_si.items():
                            self.F[col * self.m + j, col_ * self.m + j_] += weight * full_dict_with_QVQ_for_all_X[col, col_] * weight_
        # ~~~~~~~~~ GET F and g_bold end

        # ~~~~~~~~~ GET h
        self.h = 0

        for index, y in enumerate(self.Y):
            self.h += y.T.dot(self.Q.T).dot(self.V[index]).dot(self.Q).dot(y)

        # ~~~~~~~~~ GET h end

    def get_S(self) -> None:

        # ~~~~~~~~~ GET S
        self.S = np.zeros((3 * self.m, 3 * self.m))

        # TODO in case of not uniform this should be checked

        # alpha_ck = 1 / self.l_r
        alpha_ck = {}
        for col, grid_coordinates in zip(range(3), [self.space.r_grid_coordinates,
                                                    self.space.g_grid_coordinates,
                                                    self.space.b_grid_coordinates]):
            for k in range(len(grid_coordinates) - 1):
                alpha_ck[(col, k)] = grid_coordinates[k + 1] - grid_coordinates[k]

        j_r = 0
        j_g = 0
        j_b = 0
        for j in range(1, self.m - 1):
            j_r += 1
            if j_r > self.l_r:
                j_r = 0
                j_g += 1
            if j_g > self.l_g:
                j_g = 0
                j_b += 1
            j_c = [j_r, j_g, j_b]
            for col in range(3):
                hasneighbours = False
                if col == 0 and (1 <= j_r <= self.l_r - 1):
                    hasneighbours = True
                    j_minus = j - 1
                    j_plus = j + 1
                elif col == 1 and (1 <= j_g <= self.l_g - 1):
                    hasneighbours = True
                    j_minus = j - self.l_r - 1
                    j_plus = j + self.l_r + 1

                elif col == 2 and (1 <= j_b <= self.l_b - 1):
                    hasneighbours = True
                    j_minus = j - (self.l_r + 1) * (self.l_g + 1)
                    j_plus = j + (self.l_r + 1) * (self.l_g + 1)

                if hasneighbours:
                    for col_ in range(3):
                        self.S[col_ * self.m + j, col_ * self.m + j] += \
                            (alpha_ck[(col, j_c[col] - 1)] ** (-1) + alpha_ck[(col, j_c[col])] ** (-1)) ** 2

                        self.S[col_ * self.m + j_minus, col_ * self.m + j_minus] += \
                            (alpha_ck[(col, j_c[col] - 1)] ** (-2))

                        self.S[col_ * self.m + j_plus, col_ * self.m + j_plus] += (alpha_ck[(col, j_c[col])] ** (-2))

                        # not diag

                        self.S[col_ * self.m + j, col_ * self.m + j_minus] -= \
                            (alpha_ck[(col, j_c[col] - 1)] ** (-1) + alpha_ck[(col, j_c[col])] ** (-1)) / \
                            alpha_ck[(col, j_c[col] - 1)]

                        self.S[col_ * self.m + j_minus, col_ * self.m + j] -= \
                            (alpha_ck[(col, j_c[col] - 1)] ** (-1) + alpha_ck[(col, j_c[col])] ** (-1)) / \
                            alpha_ck[(col, j_c[col] - 1)]

                        self.S[col_ * self.m + j, col_ * self.m + j_plus] -= \
                            (alpha_ck[(col, j_c[col] - 1)] ** (-1) + alpha_ck[(col, j_c[col])] ** (-1)) / \
                            alpha_ck[(col, j_c[col])]

                        self.S[col_ * self.m + j_plus, col_ * self.m + j] -= \
                            (alpha_ck[(col, j_c[col] - 1)] ** (-1) + alpha_ck[(col, j_c[col])] ** (-1)) / \
                            alpha_ck[(col, j_c[col])]

                        self.S[col_ * self.m + j_minus, col_ * self.m + j_plus] += \
                            (alpha_ck[(col, j_c[col] - 1)] * alpha_ck[(col, j_c[col])]) ** (-1)
                        self.S[col_ * self.m + j_plus, col_ * self.m + j_minus] += \
                            (alpha_ck[(col, j_c[col] - 1)] * alpha_ck[(col, j_c[col])]) ** (-1)

        # ~~~~~~~~~ GET S end

    def get_K(self) -> None:
        # ~~~~~~~~~ GET K
        self.K = np.zeros((3 * self.m, 3 * self.m))
        j_dict_keys = ['000', '001', '010', '011',
                       '100', '101', '110', '111']
        K_tilda = {}
        K_tilda_check = np.zeros((len(j_dict_keys), len(j_dict_keys)))

        for i, row in enumerate(j_dict_keys):
            for j, col in enumerate(j_dict_keys):
                K_tilda_check[i, j] = -1
                K_tilda[row + col] = -1

                if row == col:
                    K_tilda[row + col] = 3
                    K_tilda_check[i, j] = 3

                if (int('0b' + row, 2) + int('0b' + col, 2)) == 7:
                    K_tilda[row + col] = 3
                    K_tilda_check[i, j] = 3

        j_dict = {}

        for j_b in range(self.l_b):
            for j_g in range(self.l_g):
                for j_r in range(self.l_r):

                    j_dict['000'] = j_r + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b
                    j_dict['001'] = j_r + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * (j_b + 1)
                    j_dict['010'] = j_r + (self.l_r + 1) * (j_g + 1) + (self.l_r + 1) * (self.l_g + 1) * j_b
                    j_dict['011'] = j_r + (self.l_r + 1) * (j_g + 1) + (self.l_r + 1) * (self.l_g + 1) * (j_b + 1)
                    j_dict['100'] = j_r + 1 + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b
                    j_dict['101'] = j_r + 1 + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * (j_b + 1)
                    j_dict['110'] = j_r + 1 + (self.l_r + 1) * (j_g + 1) + (self.l_r + 1) * (self.l_g + 1) * j_b
                    j_dict['111'] = j_r + 1 + (self.l_r + 1) * (j_g + 1) + (self.l_r + 1) * (self.l_g + 1) * (j_b + 1)
                    for col in range(3):
                        for p in j_dict_keys:
                            for q in j_dict_keys:
                                self.K[col * self.m + j_dict[p], col * self.m + j_dict[q]] += K_tilda[p + q]
        # ~~~~~~~~~ GET K end

    def get_R_and_s_bold_and_t(self):
        # ~~~~~~~~~ GET R and s bold. NOTE: CASE RGB->XYZ
        self.R = np.zeros((3 * self.m, 3 * self.m))
        self.s_bold = np.zeros((3 * self.m))

        p_bold_R = np.array([[0.488718, 0.176204, 0.0]])
        p_bold_G = np.array([[0.31068, 0.812985, 0.0102048]])
        p_bold_B = np.array([[0.200602, 0.0108109, 0.989795]])

        n_bold_R = np.cross(p_bold_G, p_bold_B).T  # TODO check if .T should be here
        n_bold_G = np.cross(p_bold_B, p_bold_R).T
        n_bold_B = np.cross(p_bold_R, p_bold_G).T

        N_tilde_col = {}
        for col in ['R', 'G', 'B']:
            N_tilde_col[col] = eval(f'n_bold_{col}.dot(n_bold_{col}.T)')

        N_col = {}
        r_bold_col = {}
        for col in ['R', 'G', 'B']:
            nu = 1 / (eval(f'n_bold_{col}.flatten().dot(n_bold_{col}.flatten())'))
            N_col[col] = nu * N_tilde_col[col]
            r_bold_col[col] = N_col[col].dot(eval(f'p_bold_{col}.T'))

        for j_g in range(self.l_g + 1):
            for j_b in range(self.l_b + 1):
                cur_j = (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b
                j_ = cur_j + self.l_r
                for color in range(3):
                    self.s_bold[color * self.m + j_] -= self.omega_weights['R1'] * r_bold_col['R'][color]
                    for color_ in range(3):
                        self.R[color * self.m + cur_j, color_ * self.m + cur_j] += \
                            self.omega_weights['R0'] * N_col['R'][color, color_]

                        self.R[color * self.m + j_, color_ * self.m + j_] += \
                            self.omega_weights['R1'] * N_col['R'][color, color_]

        for j_r in range(self.l_r + 1):
            for j_b in range(self.l_b + 1):
                cur_j = j_r + (self.l_r + 1) * (self.l_g + 1) * j_b
                j_ = cur_j + (self.l_r + 1) * self.l_g
                for color in range(3):
                    self.s_bold[color * self.m + j_] -= self.omega_weights['G1'] * r_bold_col['G'][color]
                    for color_ in range(3):
                        self.R[color * self.m + cur_j, color_ * self.m + cur_j] += \
                            self.omega_weights['G0'] * N_col['G'][color, color_]

                        self.R[color * self.m + j_, color_ * self.m + j_] += \
                            self.omega_weights['G1'] * N_col['G'][color, color_]

        for j_r in range(self.l_r + 1):
            for j_g in range(self.l_g + 1):
                cur_j = j_r + (self.l_r + 1) * j_g
                j_ = cur_j + (self.l_r + 1) * (self.l_g + 1) * self.l_b
                for color in range(3):
                    self.s_bold[color * self.m + j_] -= self.omega_weights['B1'] * r_bold_col['B'][color]
                    for color_ in range(3):
                        self.R[color * self.m + cur_j, color_ * self.m + cur_j] += \
                            self.omega_weights['B0'] * N_col['B'][color, color_]

                        self.R[color * self.m + j_, color_ * self.m + j_] += \
                            self.omega_weights['B1'] * N_col['B'][color, color_]

        self.t = self.omega_weights['R1'] * (self.l_g + 1) * (self.l_b + 1) * p_bold_R.dot(r_bold_col['R']) + \
                 self.omega_weights['G1'] * (self.l_r + 1) * (self.l_b + 1) * p_bold_G.dot(r_bold_col['G']) + \
                 self.omega_weights['B1'] * (self.l_r + 1) * (self.l_g + 1) * p_bold_B.dot(r_bold_col['B'])

        # ~~~~~~~~~END  GET R and s bold. NOTE: CASE RGB->XYZ

    def get_grid_from_vector(self, vector: np.array) -> dict:
        assert len(vector) == 3 * self.m
        grid = {}
        for j in range(self.m):
            cur_point = []
            for col in range(3):
                cur_point.append(vector[col * self.m + j])
            grid[j] = cur_point

        return grid

    @timing
    def calculate(self):
        self.get_F_and_g_bold_and_h()
        self.get_S()
        self.get_K()
        self.get_R_and_s_bold_and_t()
        self.check_symmetric()

    @timing
    def get_optimal_b(self, lambda_S: float, lambda_K: float, lambda_R: float, lambda_contrast: float = 0) -> np.array:
        b_opt = numba_get_optimal_b(F=self.F, S=self.S, K=self.K, R=self.R,
                                    g_bold=self.g_bold, s_bold=self.s_bold,
                                    lambda_S=lambda_S,
                                    lambda_K=lambda_K,
                                    lambda_R=lambda_R,
                                    lambda_contrast=lambda_contrast,
                                    T_contrast=self.T_contrast,
                                    u_contrast=self.u_contrast
                                    )
        return b_opt

    @timing
    def get_E_optimal(self, lambda_S: float, lambda_K: float, lambda_R: float) -> np.array:
        E_optimal = self.h + lambda_R * self.t - ((self.g_bold - lambda_R * self.s_bold).T.dot(
            np.linalg.inv(self.F + lambda_S * self.S + lambda_K * self.K + lambda_R * self.R)).dot(
            self.g_bold - lambda_R * self.s_bold))
        return E_optimal

    @timing
    def get_E_color(self, b_opt: np.array) -> float:

        E_color = b_opt.T.dot(self.F).dot(b_opt) - 2 * self.g_bold.T.dot(b_opt) + self.h
        return E_color

    @timing
    def get_E_reg(self, lambda_S: float, lambda_K: float, lambda_R: float, b_opt: np.array) -> float:

        E_reg = b_opt.T.dot((lambda_S * self.S + lambda_K * self.K + lambda_R * self.R)).dot(
            b_opt) + 2 * lambda_R * self.s_bold.T.dot(b_opt) + lambda_R * self.t
        return E_reg

    @timing
    def get_E_contrast(self, b_opt: np.array, lambda_contrast: float) -> float:

        E_contrast = b_opt.T.dot(self.T_contrast).dot(b_opt) + 2 * self.u_contrast.T.dot(b_opt) + self.v_contrast
        return lambda_contrast * E_contrast

    def get_all_results(self) -> Tuple[np.array, np.array, Optional[plt.Axes], float, float, float, Optional[float]]:
        if not self.trained:
            self.b_opt = self.get_optimal_b(self.lambda_S, self.lambda_K, self.lambda_R, self.lambda_contrast)
            self.trained = True

        E_opt = self.get_E_optimal(self.lambda_S, self.lambda_K, self.lambda_R)

        E_color = float(self.get_E_color(self.b_opt))
        E_reg = float(self.get_E_reg(self.lambda_S, self.lambda_K, self.lambda_R, self.b_opt))
        if self.lambda_contrast:
            E_contrast = float(self.get_E_contrast(self.b_opt, self.lambda_contrast))
        self.b_opt = self.get_grid_from_vector(self.b_opt)
        if self.plot_graph:
            points_image = plot_dict(self.b_opt)
        else:
            points_image = None

        if self.lambda_contrast:
            return np.array(self.space.full_space), np.array(list(self.b_opt.values())), points_image, float(
                E_opt), float(E_color), float(E_reg), float(E_contrast)

        return np.array(self.space.full_space), np.array(list(self.b_opt.values())), points_image, float(E_opt), float(E_color), float(E_reg)

    def predict(self, X: np.array) -> np.array:
        assert self.trained, "Should be trained before making predictions"
        pred = []
        for key, x in enumerate(X):
            res = [0, 0, 0]
            _, cur_interpolation_results = self.interpolation.get_weights_for_full_grid(x)
            for kw, vw in cur_interpolation_results.items():
                for i in range(3):
                    res[i] += vw * self.b_opt[kw][i]
            pred.append(res)
        return np.array(pred)

    @timing
    def check_symmetric(self) -> None:
        for mat in [self.F, self.K, self.S, self.R]:
            assert check_symmetric(mat), f'Matrix {mat} is not symmetric'

    def get_A(self):
        A = np.zeros((3, self.m))
        for j_r in range(self.l_r + 1):
            for j_g in range(self.l_g + 1):
                for j_b in range(self.l_b + 1):
                    A[0, j_r + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b] = j_r / self.l_r
                    A[1, j_r + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b] = j_g / self.l_g
                    A[2, j_r + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b] = j_b / self.l_b
        return A

    def get_r_A_base(self) -> np.array:
        A = self.get_A()
        b_opt = self.get_optimal_b(self.lambda_S, self.lambda_K, self.lambda_R, lambda_contrast=0)
        b_opt = self.get_grid_from_vector(b_opt)
        B0 = np.zeros((3, self.m))
        for j_r in range(self.l_r + 1):
            for j_g in range(self.l_g + 1):
                for j_b in range(self.l_b + 1):
                    cur_point = b_opt[self.space.get_j_from_jr_jg_rb(j_r, j_g, j_b)]
                    B0[0, j_r + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b] = cur_point[0]
                    B0[1, j_r + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b] = cur_point[1]
                    B0[2, j_r + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b] = cur_point[2]

        bary_center_a = np.array([[0.5, 0.5, 0.5]]).reshape(3, 1)
        bary_center_b = ((1 / self.m) * (B0.sum(axis=1))).reshape(3, 1)

        D = (A - bary_center_a).dot((B0 - bary_center_b).T)  # TODO check reshape
        svd_target = scipy.linalg.sqrtm(self.Gram).dot(D).dot(scipy.linalg.sqrtm(self.Gram))

        U, s, V_T = np.linalg.svd(svd_target, full_matrices=True)

        R_tilda = V_T.T.dot(U.T)

        R = np.linalg.inv(scipy.linalg.sqrtm(self.Gram)).dot(R_tilda).dot(np.linalg.inv(scipy.linalg.sqrtm(self.Gram)))
        z = bary_center_b - R.dot(bary_center_a)
        return R.dot(A) + z

    def get_r_A(self) -> np.array:
        intermediate_res = self.get_r_A_base()
        return np.linalg.inv(scipy.linalg.sqrtm(self.Gram_y)).dot(scipy.linalg.sqrtm(self.Gram)).dot(intermediate_res)

    def get_T_u_v_for_contrast(self):

        xx = self.xx_0 * max(np.linalg.norm(self.image_X, axis=-1).flatten())

        Nu = self.image_X.shape[0]  # <- rows
        Xi = self.image_X.shape[1]  # <- columns
        Gram = np.identity(3)

        A = self.get_r_A()

        ga = Gram.dot(A)
        aga = A.T.dot(ga)
        all_weights_for_all_colors = {}

        for xi in range(Xi):
            for eta in range(Nu):
                if all_weights_for_all_colors.get((xi, eta), None) is None:
                    if sigma_contrast(self.image_X[xi, eta]) > 0:
                        _, all_weights_for_all_colors[(xi, eta)] = self.interpolation.get_weights_for_full_grid(
                            self.image_X[xi, eta])
                        if eta - 1 >= 0:
                            if all_weights_for_all_colors.get((xi, eta - 1), None) is None:
                                _, all_weights_for_all_colors[
                                    (xi, eta - 1)] = self.interpolation.get_weights_for_full_grid(
                                    self.image_X[xi, eta - 1])

                        else:
                            if all_weights_for_all_colors.get((xi, eta + 1), None) is None:
                                _, all_weights_for_all_colors[
                                    (xi, eta + 1)] = self.interpolation.get_weights_for_full_grid(
                                    self.image_X[xi, eta + 1])

                        if xi - 1 >= 0:
                            if all_weights_for_all_colors.get((xi - 1, eta), None) is None:
                                _, all_weights_for_all_colors[
                                    (xi - 1, eta)] = self.interpolation.get_weights_for_full_grid(
                                    self.image_X[xi - 1, eta])

                        else:
                            if all_weights_for_all_colors.get((xi + 1, eta), None) is None:
                                _, all_weights_for_all_colors[
                                    (xi + 1, eta)] = self.interpolation.get_weights_for_full_grid(
                                    self.image_X[xi + 1, eta])

        for eta in range(Nu):
            for xi in range(Xi):

                point_factor = sigma_contrast(self.image_X[xi, eta])

                if point_factor > 0:

                    ww_0 = all_weights_for_all_colors[(xi, eta)]
                    Q_pretty_0 = ww_0.keys()

                    wweta = all_weights_for_all_colors.get((xi, eta - 1),
                                                           all_weights_for_all_colors.get((xi, eta + 1), None))
                    Q_pretty_eta = wweta.keys()

                    wwxi = all_weights_for_all_colors.get((xi - 1, eta),
                                                          all_weights_for_all_colors.get((xi + 1, eta), None))
                    Q_pretty_xi = wwxi.keys()

                    Q_pretty = set(Q_pretty_0).union(set(Q_pretty_xi)).union(set(Q_pretty_eta))

                    delta_W = np.zeros((len(Q_pretty), 2))
                    Q_index_map = {value: index for index, value in enumerate(Q_pretty)}

                    for j_real in Q_pretty_0:
                        j = Q_index_map[j_real]
                        delta_W[j, 0] = ww_0[j_real]
                        delta_W[j, 1] = ww_0[j_real]

                    for j_real in Q_pretty_xi:
                        j = Q_index_map[j_real]
                        delta_W[j, 0] -= wwxi[j_real]

                    for j_real in Q_pretty_eta:
                        j = Q_index_map[j_real]
                        delta_W[j, 1] -= wweta[j_real]

                    ww = delta_W.dot(delta_W.T)

                    # FIXME slow slicing
                    aga_slice = np.zeros((len(Q_pretty), len(Q_pretty)))
                    for q1 in Q_pretty:
                        for q2 in Q_pretty:
                            aga_slice[Q_index_map[q1], Q_index_map[q2]] = aga[q1, q2]

                    Cxnorm2 = 0
                    wwaga = ww.dot(aga_slice)
                    agawwaga = aga_slice.dot(wwaga)

                    for j_real in set(Q_pretty_0).union(set(Q_pretty_xi)):
                        for j_real_ in set(Q_pretty_0).union(set(Q_pretty_xi)):
                            j = Q_index_map[j_real]
                            j_ = Q_index_map[j_real_]
                            Cxnorm2 += delta_W[j, 0] * agawwaga[j, j_] * delta_W[j_, 0]

                    for j_real in set(Q_pretty_0).union(set(Q_pretty_eta)):
                        for j_real_ in set(Q_pretty_0).union(set(Q_pretty_eta)):
                            j = Q_index_map[j_real]
                            j_ = Q_index_map[j_real_]
                            Cxnorm2 += delta_W[j, 1] * agawwaga[j, j_] * delta_W[j_, 1]

                    self.v_contrast += 4 * point_factor * Cxnorm2 / (Cxnorm2 + xx ** 2)

                    wwagaw = wwaga.dot(delta_W)
                    # FIXME slow slicing
                    ga_slice = np.zeros((3, len(Q_pretty)))
                    for q1 in set(Q_pretty):
                        for col in range(3):
                            ga_slice[col, Q_index_map[q1]] = ga[col, q1]

                    gawwagaw = ga_slice.dot(wwagaw)
                    gaw = ga_slice.dot(delta_W)
                    for q1 in Q_pretty:
                        for col in range(3):
                            uu = 0.0
                            for i in range(2):
                                uu += delta_W[Q_index_map[q1], i] * gawwagaw[col, i]
                                uu += gaw[col, i] * wwagaw[Q_index_map[q1], i]

                            self.u_contrast[col * self.m + q1] -= 2 * point_factor * uu / (Cxnorm2 + xx ** 2)

                    gawwag = gaw.dot(gaw.T)
                    gaww = ga_slice.dot(ww)
                    for q1 in Q_pretty:
                        for col in range(3):
                            for q1_ in Q_pretty:
                                for col_ in range(3):
                                    t = 0.0
                                    for i in range(2):
                                        t += delta_W[Q_index_map[q1], i] * gawwag[col, col_] * delta_W[
                                            Q_index_map[q1_], i]
                                        t += delta_W[Q_index_map[q1], i] * gaww[col, Q_index_map[q1_]] * gaw[col_, i]
                                        t += delta_W[Q_index_map[q1_], i] * gaww[col_, Q_index_map[q1]] * gaw[col, i]
                                        t += gaw[col, i] * ww[Q_index_map[q1], Q_index_map[q1_]] * gaw[col_, i]

                                    self.T_contrast[col * self.m + q1, col_ * self.m + q1_] += point_factor * t / (
                                            Cxnorm2 + xx ** 2)


                else:  # (point_factor <= 0)
                    pass


class UniformGridSearchFromMatricesRGB(UniformGridSearchFromMatrices):

    def get_F_and_g_bold_and_h(self) -> None:
        self.get_V()

        for index, cur_train_point in enumerate(self.X):
            weights_for_grid, weights_for_grid_si = self.interpolation.get_weights_for_full_grid(cur_train_point)
            self.full_dict_with_weights_for_all_X[index] = weights_for_grid_si

        # ~~~~~~~~~ GET F and g_bold
        self.F = np.zeros((3 * self.m, 3 * self.m))
        self.g_bold = np.zeros((3 * self.m))
        for index, cur_train_point in enumerate(self.X):
            _, weights_for_grid_si = self.interpolation.get_weights_for_full_grid(cur_train_point)

            for col in range(3):
                for j, weight in weights_for_grid_si.items():
                    self.g_bold[col * self.m + j] += weight * self.Y[index, col]
                    for j_, weight_ in weights_for_grid_si.items():
                        self.F[col * self.m + j, col * self.m + j_] += weight * weight_
        # ~~~~~~~~~ GET F and g_bold END

        # ~~~~~~~~~ GET h
        self.h = 0

        for y in self.Y:
            self.h += y.T.dot(y)

        # ~~~~~~~~~ GET h end

    def get_R_and_s_bold_and_t(self):
        # ~~~~~~~~~ GET R and s bold. NOTE: CASE RGB->RGB
        self.R = np.zeros((3 * self.m, 3 * self.m))
        self.s_bold = np.zeros((3 * self.m))

        for j_g in range(self.l_g + 1):
            for j_b in range(self.l_b + 1):
                cur_j = (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b
                j_ = cur_j + self.l_r
                col = 0  # X
                self.s_bold[col * self.m + j_] -= self.omega_weights['R1']
                self.R[col * self.m + cur_j, col * self.m + cur_j] += self.omega_weights['R0']
                self.R[col * self.m + j_, col * self.m + j_] += self.omega_weights['R1']

        for j_r in range(self.l_r + 1):
            for j_b in range(self.l_b + 1):
                cur_j = j_r + (self.l_r + 1) * (self.l_g + 1) * j_b
                j_ = cur_j + (self.l_r + 1) * self.l_g
                col = 1  # Y
                self.s_bold[col * self.m + j_] -= self.omega_weights['G1']
                self.R[col * self.m + cur_j, col * self.m + cur_j] += self.omega_weights['G0']
                self.R[col * self.m + j_, col * self.m + j_] += self.omega_weights['G1']

        for j_r in range(self.l_r + 1):
            for j_g in range(self.l_g + 1):
                cur_j = j_r + (self.l_r + 1) * j_g
                j_ = cur_j + (self.l_r + 1) * (self.l_g + 1) * self.l_b
                col = 2  # Z
                self.s_bold[col * self.m + j_] -= self.omega_weights['B1']
                self.R[col * self.m + cur_j, col * self.m + cur_j] += self.omega_weights['B0']
                self.R[col * self.m + j_, col * self.m + j_] += self.omega_weights['B1']

        for color in range(3):
            self.R[color * self.m, color * self.m] += self.omega_weights['000']

        self.t = self.omega_weights['R1'] * (self.l_g + 1) * (self.l_b + 1) + \
                 self.omega_weights['G1'] * (self.l_r + 1) * (self.l_b + 1) + \
                 self.omega_weights['B1'] * (self.l_r + 1) * (self.l_g + 1)

        # ~~~~~~~~~END  GET R and s bold. NOTE: CASE RGB->RGB

    def get_r_A(self) -> np.array:
        return self.get_r_A_base()


class UniformGridSearchFromMatricesProlab(UniformGridSearchFromMatrices):

    def get_V_with_lid(self):
        # FIXME check this because still not clear how to construct based on j?
        v_l = 1
        v_a = 1
        v_b = 1

        V_with_lid = [get_diagonal_matrix_with_vals(v_l, v_a, v_b)] * self.m
        V_with_lid = np.array(V_with_lid)
        return V_with_lid

    def get_R_and_s_bold_and_t(self):
        self.R = np.zeros((3 * self.m, 3 * self.m))
        self.s_bold = np.zeros((3 * self.m))

        V_with_lid = self.get_V_with_lid()

        # ~~~~~~~~~ GET R and s bold. NOTE: CASE Gamut
        p_bold_R = np.array([[0.48657095], [0.22897456], [0.0]])  # FIXME should be same as in parent method
        p_bold_G = np.array([[0.26566769], [0.69173852], [0.04511338]])
        p_bold_B = np.array([[0.19821729], [0.07928691], [1.04394437]])
        P_Y = np.array([p_bold_R, p_bold_G, p_bold_B]).squeeze()
        q_bold = np.array([[0.7554], [3.8666], [1.6739]])  # local q_bold

        U = self.Q.dot(P_Y)
        u_bold_r = U[:, 0]
        u_bold_g = U[:, 1]
        u_bold_b = U[:, 2]
        v_bold_T = q_bold.T.dot(P_Y)  # local q_bold

        v_r, v_g, v_b = v_bold_T.squeeze()

        n_bold_R_0 = np.cross(u_bold_g, u_bold_b)
        n_bold_G_0 = np.cross(u_bold_b, u_bold_r)
        n_bold_B_0 = np.cross(u_bold_r, u_bold_g)

        n_bold_R_1 = np.cross((1 + v_r) * u_bold_g - v_g * u_bold_r, (1 + v_r) * u_bold_b - v_b * u_bold_r)
        n_bold_G_1 = np.cross((1 + v_g) * u_bold_b - v_b * u_bold_g, (1 + v_g) * u_bold_r - v_r * u_bold_g)
        n_bold_B_1 = np.cross((1 + v_b) * u_bold_r - v_r * u_bold_b, (1 + v_b) * u_bold_g - v_g * u_bold_b)

        kappa = np.array([[n_bold_R_0.T.dot(n_bold_R_0), n_bold_R_1.T.dot(n_bold_R_1)],
                          [n_bold_G_0.T.dot(n_bold_G_0), n_bold_G_1.T.dot(n_bold_G_1)],
                          [n_bold_B_0.T.dot(n_bold_B_0), n_bold_B_1.T.dot(n_bold_B_1)]])

        omega_bold = np.array([[n_bold_R_0.T.dot(U), n_bold_R_1.T.dot(U)],
                               [n_bold_G_0.T.dot(U), n_bold_G_1.T.dot(U)],
                               [n_bold_B_0.T.dot(U), n_bold_B_1.T.dot(U)]])

        # kappa: [col in {R,G,B}, nu in {0,1}]
        # omega_bold: [col in {R,G,B}, nu in {0,1}, col in {R,G,B}]

        alpha_0 = 1
        alpha_1 = 1
        t_R = omega_bold[0, 1, 0] / (1 + v_r) ** 2 / kappa[0, 1]  # TODO what is it?
        self.t = 0  # TODO check initialization here

        # R
        for j_g in range(self.l_g + 1):
            for j_b in range(self.l_b + 1):
                cur_j = (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b
                j_ = cur_j + self.l_r

                if self.weighted:
                    alpha_0 = n_bold_R_0.T.dot(V_with_lid[cur_j]).dot(n_bold_R_0) / kappa[0, 0]
                    alpha_1 = n_bold_R_1.T.dot(V_with_lid[j_]).dot(n_bold_R_1) / kappa[0, 1]
                    self.t += alpha_1 * (omega_bold[0, 1, 0] ** 2) / (1 + v_r) ** 2 / kappa[0, 1]

                for color in range(3):
                    self.s_bold[color * self.m + j_] -= alpha_1 * omega_bold[0, 1, 0] * omega_bold[0, 1, color] / (1 + v_r) / (1 + v_bold_T.dot(self.space.all_vertices_single[j_])) / kappa[0, 1]

                    for color_ in range(3):
                        self.R[color * self.m + cur_j, color_ * self.m + cur_j] += alpha_0 * omega_bold[0, 0, color] * omega_bold[0, 0, color_] / (1 + v_bold_T.dot(self.space.all_vertices_single[cur_j])) ** 2 / kappa[0, 0]
                        self.R[color * self.m + j_, color_ * self.m + j_] += alpha_1 * omega_bold[0, 1, color] * omega_bold[0, 1, color_] / (1 + v_bold_T.dot(self.space.all_vertices_single[j_])) ** 2 / kappa[0, 1]

        # G
        for j_r in range(self.l_r + 1):
            for j_b in range(self.l_b + 1):
                cur_j = j_r + (self.l_r + 1) * (self.l_g + 1) * j_b
                j_ = cur_j + (self.l_r + 1) * self.l_g

                if self.weighted:
                    alpha_0 = n_bold_G_0.T.dot(V_with_lid[cur_j]).dot(n_bold_G_0) / kappa[1, 0]
                    alpha_1 = n_bold_G_1.T.dot(V_with_lid[j_]).dot(n_bold_G_1) / kappa[1, 1]
                    self.t += alpha_1 * (omega_bold[1, 1, 1] ** 2) / (1 + v_g) ** 2 / kappa[1, 1]

                for color in range(3):
                    self.s_bold[color * self.m + j_] -= alpha_1 * omega_bold[1, 1, 1] * omega_bold[1, 1, color] / (1 + v_g) / (1 + v_bold_T.dot(self.space.all_vertices_single[j_])) / kappa[1, 1]

                    for color_ in range(3):
                        self.R[color * self.m + cur_j, color_ * self.m + cur_j] += alpha_0 * omega_bold[1, 0, color] * omega_bold[1, 0, color_] / (1 + v_bold_T.dot(self.space.all_vertices_single[cur_j])) ** 2 / kappa[1, 0]
                        self.R[color * self.m + j_, color_ * self.m + j_] += alpha_1 * omega_bold[1, 1, color] * omega_bold[1, 1, color_] / (1 + v_bold_T.dot(self.space.all_vertices_single[j_])) ** 2 / kappa[1, 1]

        # B
        for j_r in range(self.l_r + 1):
            for j_g in range(self.l_g + 1):
                cur_j = j_r + (self.l_r + 1) * j_g
                j_ = cur_j + (self.l_r + 1) * (self.l_g + 1) * self.l_b

                if self.weighted:
                    alpha_0 = n_bold_B_0.T.dot(V_with_lid[cur_j]).dot(n_bold_B_0) / kappa[2, 0]
                    alpha_1 = n_bold_B_1.T.dot(V_with_lid[j_]).dot(n_bold_B_1) / kappa[2, 1]
                    self.t += alpha_1 * (omega_bold[2, 1, 2] ** 2) / (1 + v_b) ** 2 / kappa[2, 1]

                for color in range(3):
                    self.s_bold[color * self.m + j_] -= alpha_1 * omega_bold[2, 1, 2] * omega_bold[2, 1, color] / (1 + v_b) / (1 + v_bold_T.dot(self.space.all_vertices_single[j_])) / kappa[2, 1]

                    for color_ in range(3):
                        self.R[color * self.m + cur_j, color_ * self.m + cur_j] += alpha_0 * omega_bold[2, 0, color] * omega_bold[2, 0, color_] / (1 + v_bold_T.dot(self.space.all_vertices_single[cur_j])) ** 2 / kappa[2, 0]
                        self.R[color * self.m + j_, color_ * self.m + j_] += alpha_1 * omega_bold[2, 1, color] * omega_bold[2, 1, color_] / (1 + v_bold_T.dot(self.space.all_vertices_single[j_])) ** 2 / kappa[2, 1]

        if not self.weighted:
            self.t = (self.l_g + 1) * (self.l_b + 1) * (omega_bold[0, 1, 0] ** 2) / (((1 + v_r) ** 2) * kappa[0, 1]) + \
                     (self.l_r + 1) * (self.l_b + 1) * (omega_bold[1, 1, 1] ** 2) / (((1 + v_g) ** 2) * kappa[1, 1]) + \
                     (self.l_r + 1) * (self.l_g + 1) * (omega_bold[2, 1, 2] ** 2) / (((1 + v_b) ** 2) * kappa[2, 1])

    def get_r_A(self):
        return self.get_A()


task_mappings = {
    'RGB2XYZ': UniformGridSearchFromMatrices,
    'RGB2RGB': UniformGridSearchFromMatricesRGB,
    'GamutInProlab': UniformGridSearchFromMatricesProlab
}

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    cur_experiment = 'RGB'
    experiments = {
        
        
        
       
        'GT': ['GT_points_rgb.csv', 'GT_points_xyz.csv'],
        'GT1': ['GT_points_rgb.csv', 'GT_points_rgb.csv'],
        'GT2': ['GT_8.csv', 'GT_8.csv'],
    }

    prolab_function = get_proper_prolab_function(cur_experiment)
    # sys.stdout = open(f'error_log_{cur_experiment}.txt', 'w')
    path_to_rgb, path_to_xyz_real = experiments[cur_experiment][:2]

    X = np.genfromtxt(path_to_rgb, delimiter=',')
    Y = np.genfromtxt(path_to_xyz_real, delimiter=',')

    l_S = None  # 0.0001
    l_R = None
    l_K = None

    o_w = {
        'R0': 1,
        'R1': 1,
        'G0': 1,
        'G1': 1,
        'B0': 1,
        'B1': 1,
        '000': 0
    }

    for num_fixed in range(4, 5):
        l = {
            'r': num_fixed,
            'g': num_fixed,
            'b': num_fixed
        }
        print(f'~~~~ Number of points is {l} ~~~~')
        space = UniformSearchSpace(l['r'] + 1, l['g'] + 1, l['b'] + 1)
        trilinear_int = TrilinearInterpolation(space)
        import cv2

        img1 = cv2.imread('1.png')
        img1 = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        mats = UniformGridSearchFromMatricesProlab(space, trilinear_int, X, Y,
                                                   lambda_S=l_S,
                                                   lambda_R=l_R,
                                                   lambda_K=l_K,
                                                   omega_weights=o_w, delta_for_lambda=0.01, weighted=True,
                                                   calculate_lambda_with_grid_and_train_size=True,
                                                   lambda_contrast=1,
                                                   image_for_contrast=img1)
        a, b, c, E_opt, E_color, E_reg = mats.get_all_results()
        print(E_reg/E_color)


