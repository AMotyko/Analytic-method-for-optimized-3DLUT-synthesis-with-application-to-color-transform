from uniform_search_space import timing
import scipy
import numpy as np
import matplotlib.pyplot as plt

def plot_points(a_opt, b_opt):
    a0, a1, a2 = np.meshgrid(a_opt[0], a_opt[1], a_opt[2], indexing='ij')
    a_opt_points = np.stack((a0, a1, a2), axis=3).reshape(len(a0) * len(a1) * len(a2), 3)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121, projection='3d')
    plot_122 = fig.add_subplot(122, projection='3d')

    ax.set_title('a_opt')
    ax.scatter(a_opt_points[:, 0], a_opt_points[:, 1], a_opt_points[:, 2], zdir='z', c='red')

    plot_122.set_title('b_opt')
    plot_122.scatter(b_opt[:,0], b_opt[:,1], b_opt[:,2], zdir='z', c='red')

    plt.show()
    return fig



class NonuniformLut(object):

    def __init__(self, uniform_lut):
        self.uniform_lut = uniform_lut
        self.l_r = self.uniform_lut.number_of_nodes_in_R
        self.l_g = self.uniform_lut.number_of_nodes_in_G
        self.l_b = self.uniform_lut.number_of_nodes_in_B

    def recalculate_value(self, alpha_c, X_c):
        alpha_c_mod = np.insert(alpha_c, 0, 0)
        alpha_c_mod = np.insert(alpha_c_mod, len(alpha_c_mod), 1)
        l_c = len(alpha_c_mod) - 1


        for j in range(1, len(alpha_c_mod)):
            if X_c <= alpha_c_mod[j]:
                return j / l_c + (X_c - alpha_c_mod[j]) / (l_c * (alpha_c_mod[j] - alpha_c_mod[j - 1]))
        return j / l_c + (X_c - 1) / (l_c * (alpha_c_mod[j] - alpha_c_mod[j - 1]))

    def recalculate_train(self, alpha, X):
        alpha_r = alpha[0: self.l_r - 1]
        alpha_g = alpha[self.l_r - 1: self.l_r + self.l_g - 2]
        alpha_b = alpha[self.l_r + self.l_g - 2: self.l_r + self.l_g + self.l_b - 3]
        X_transformed = np.zeros(X.shape)
        for i, val in enumerate(X):
            X_transformed[i] = np.array([self.recalculate_value(alpha_r, val[0]),
                                         self.recalculate_value(alpha_g, val[1]),
                                         self.recalculate_value(alpha_b, val[2])])
        return X_transformed

    def train(self, X, Y):
        def loss_func(alpha):
            X_transformed = self.recalculate_train(alpha, X)
            a_opt, b_opt, image, E_opt, _, _ = self.uniform_lut.train(X_transformed, Y, plot_graph = False)

            for vec in np.c_[np.zeros(3), alpha.reshape(3, len(alpha) // 3), np.ones(3)]:
                for i in range(1, len(vec)):
                    if vec[i] <= vec[i-1]:
                        E_opt += 0.2

            print(E_opt)

            return E_opt

        def loss_grad(x):
            return scipy.optimize.approx_fprime(np.array(x), loss_func, epsilon=1e-8)

        alpha_init = np.vstack((self.uniform_lut.space.r_grid_coordinates[1:-1],
                                self.uniform_lut.space.g_grid_coordinates[1:-1],
                                self.uniform_lut.space.b_grid_coordinates[1:-1]))

        print('alpha init: ', alpha_init)
        res = scipy.optimize.fmin_bfgs(loss_func, alpha_init.flatten(), fprime=loss_grad, gtol = 0.01)

        X_transformed = self.recalculate_train(res, X)
        a_opt, b_opt, image, E_opt,E_color, E_reg = self.uniform_lut.train(X_transformed, Y)

        a_opt = np.c_[np.zeros(3), res.reshape(3, len(res) // 3), np.ones(3)]

        return a_opt, b_opt, None, E_opt, E_color, E_reg, res