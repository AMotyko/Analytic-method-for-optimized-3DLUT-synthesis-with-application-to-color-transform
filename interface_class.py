from uniform_lut_construction import task_mappings
from uniform_search_space import UniformSearchSpace, TrilinearInterpolation, TetrahedralInterpolation
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


class LUTConstructor(object):

    def __init__(self, **kwargs):

        assert all(x in kwargs.keys() for x in ['number_of_nodes_in_R',
                                                'number_of_nodes_in_G',
                                                'number_of_nodes_in_B',
                                                'omega_weights', 'mode'])

        self.__dict__.update(kwargs)
        assert self.interpolation in ('trilinear', 'tetrahedral')
        assert self.mode in ('RGB2XYZ', 'RGB2RGB', 'GamutInProlab')
        self.space = UniformSearchSpace(self.number_of_nodes_in_R + 1,
                                        self.number_of_nodes_in_G + 1,
                                        self.number_of_nodes_in_B + 1)

        if self.interpolation == 'trilinear':
            self.interpolation = TrilinearInterpolation(self.space)
        elif self.interpolation == 'tetrahedral':
            self.interpolation = TetrahedralInterpolation(self.space)

        if 'lambda_S' not in kwargs.keys():
            self.lambda_S = None

        if 'lambda_R' not in kwargs.keys():
            self.lambda_R = None

        if 'lambda_K' not in kwargs.keys():
            self.lambda_K = None

        self.current_task_class = task_mappings[self.mode]
        self.ugsfm = None

        if 'calculate_lambda_with_grid_and_train_size' not in kwargs.keys():
            self.calculate_lambda_with_grid_and_train_size = False
        if 'delta_for_lambda' not in kwargs.keys():
            self.delta_for_lambda = 0.01

        self.weighted = kwargs.get('weighted', False)

        self.lambda_contrast = kwargs.get('lambda_contrast', None)
        self.image_for_contrast = kwargs.get('image_for_contrast', None)
        self.xx_0_for_contrast = kwargs.get('xx_0_for_contrast', 0.01)
        self.gram = kwargs.get('gram', np.identity(3))
        self.gram_y = kwargs.get('gram_y', np.identity(3))

    def train(self, X: np.array, Y: np.array, plot_graph : bool = True) -> Tuple[np.array, plt.Axes, np.array]:
        """
        Train LUT table
        :param X: set X
        :param Y: set Y
        :return: (numpy array with initial grid, numpy array with optimal b, image with these b - points, E_optimal, E_color, E_reg)
        """

        self.ugsfm = self.current_task_class(self.space, self.interpolation, X, Y,
                                             lambda_S=self.lambda_S,
                                             lambda_R=self.lambda_R,
                                             lambda_K=self.lambda_K,
                                             omega_weights=self.omega_weights,
                                             calculate_lambda_with_grid_and_train_size=self.calculate_lambda_with_grid_and_train_size,
                                             delta_for_lambda=self.delta_for_lambda,
                                             weighted=self.weighted,
                                             plot_graph=plot_graph,
                                             lambda_contrast=self.lambda_contrast,
                                             image_for_contrast=self.image_for_contrast,
                                             xx_0_for_contrast=self.xx_0_for_contrast,
                                             gram=self.gram,
                                             gram_y=self.gram_y
                                             )
        return self.ugsfm.get_all_results()

    def predict(self, X_test: np.array) -> np.array:
        """
        Get predictions for given test set
        :param X_test: numpy array of shape (3, None)
        :return: numpy array of shape (3, None) with predictions by last trained LUT
        """
        return self.ugsfm.predict(X_test)


if __name__ == '__main__':
    import cv2
    img1 = cv2.imread('1.png')
    print(img1.shape)
    img1 = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    parameters_of_experiment = {
        'number_of_nodes_in_R': 3,  # l_r in paper
        'number_of_nodes_in_G': 3,  # l_g in paper
        'number_of_nodes_in_B': 3,  # l_b in paper

        'lambda_R': 0.001,
        'lambda_S': 0.01,
        'lambda_K': 0.01,
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
        'mode': 'RGB2XYZ',  # 'RGB2RGB', 'GamutInProlab', RGB2XYZ
        'interpolation': 'trilinear',  # 'tetrahedral'
        'weighted': True,

        'lambda_contrast': 1.0,
        'image_for_contrast': img1,
        'xx_0_for_contrast': 0.01,
        'gram': np.identity(3),
        'gram_y': np.identity(3)
    }

    uniform_lut = LUTConstructor(**parameters_of_experiment)
    cur_experiment = 'D65'
    experiments = {
        
        
        
        'GT': ['GT_points_rgb.csv', 'GT_points_xyz.csv'],
        'GT1': ['GT_points_rgb.csv', 'GT_points_rgb.csv'],
        'GT2': ['GT_8.csv', 'GT_8.csv'],
    }

    path_to_rgb, path_to_xyz_real = experiments[cur_experiment][:2]

    X = np.genfromtxt(path_to_rgb, delimiter=',')
    Y = np.genfromtxt(path_to_xyz_real, delimiter=',')

    A, b_opt, points_image, E_opt, E_color, E_reg, E_contrast = uniform_lut.train(X, Y)
    print(E_opt)
    print(E_contrast)
    # print(uniform_lut.predict(X))
    # plt.show()
