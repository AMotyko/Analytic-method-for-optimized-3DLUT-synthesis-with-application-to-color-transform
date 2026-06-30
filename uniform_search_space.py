import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Callable
import time
from functools import wraps
import logging


def plot_dict(dict_: dict, x: Union[tuple, list, np.array] = None, name: str = None) -> plt.Axes:
    """
    plot cube which contains the point
    :param name:
    :param dict_:
    :param x:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if x:
        ax.scatter(x[0], x[1], x[2], zdir='z', c='blue')

    for value in dict_.values():
        ax.scatter(value[0], value[1], value[2], zdir='z', c='red')
        # print(f'cube = {value}')
    if name:
        plt.savefig(name)

    plt.show()
    return ax


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        logging.debug(f'func:{f.__name__} took: {te - ts:2.4f} sec')
        return result

    return wrap


default_limits = (0, 1)


def find_right_vertex_index_for_x(x_i, grid_coordinates):
    # TODO redo the function as result = x_i * grid_step
    result = np.searchsorted(grid_coordinates, x_i)
    if result < 0:
        logging.debug(f"Warning negative index {find_right_vertex_index_for_x.__name__}")
        return 0
    elif result >= len(grid_coordinates):
        logging.debug(f"Warning too big index {find_right_vertex_index_for_x.__name__}")
        return len(grid_coordinates) - 1
    return result


def get_1d_grid(number_of_points: int, dim_limits: tuple) -> np.array:
    """

    :param dim_limits: list with borders of dimenstions
    :param number_of_points: 0 = a_0 < ... < a_number_of_points = 1
    :return:
    """
    grid = np.linspace(dim_limits[0], dim_limits[1], number_of_points)
    return grid


class UniformSearchSpace(object):
    def __init__(self, number_of_points_in_r: int, number_of_points_in_g: int, number_of_points_in_b: int,
                 limits_in_r: tuple = default_limits,
                 limits_in_g: tuple = default_limits,
                 limits_in_b: tuple = default_limits, do: bool = True) -> None:
        self.r_points_num = number_of_points_in_r
        self.g_points_num = number_of_points_in_g
        self.b_points_num = number_of_points_in_b

        self.l_r = number_of_points_in_r - 1
        self.l_g = number_of_points_in_g - 1
        self.l_b = number_of_points_in_b - 1

        self.r_grid_coordinates = get_1d_grid(number_of_points_in_r, limits_in_r)
        self.g_grid_coordinates = get_1d_grid(number_of_points_in_g, limits_in_g)
        self.b_grid_coordinates = get_1d_grid(number_of_points_in_b, limits_in_b)

        self.full_space = [self.r_grid_coordinates, self.g_grid_coordinates, self.b_grid_coordinates]
        self.all_vertices = {}
        self.all_vertices_single = {}
        self.single_to_multi_map = {}
        self.fill_grid_in_space(do)

    def fill_grid_in_space(self, do: bool) -> None:
        if not do:
            return

        for j_r in range(self.r_points_num):
            for j_g in range(self.g_points_num):
                for j_b in range(self.b_points_num):
                    cur_j = self.get_j_from_jr_jg_rb(j_r, j_g, j_b)
                    self.all_vertices[(j_r, j_g, j_b)] = [self.r_grid_coordinates[j_r],
                                                          self.g_grid_coordinates[j_g],
                                                          self.b_grid_coordinates[j_b]]
                    self.all_vertices_single[cur_j] = [self.r_grid_coordinates[j_r],
                                                       self.g_grid_coordinates[j_g],
                                                       self.b_grid_coordinates[j_b]]
                    self.single_to_multi_map[cur_j] = (j_r, j_g, j_b)

    def get_j_from_jr_jg_rb(self, j_r: int, j_g: int, j_b: int) -> int:
        return j_r + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b
        # return j_r + self.r_points_num * j_g + self.r_points_num * self.g_points_num * j_b


class Interpolation(object):
    def __init__(self, search_space: UniformSearchSpace) -> None:
        self.search_space = search_space

    def all_weights_for_cube(self, x: Union[tuple, list, np.array],
                             right_vertices: Union[tuple, list, np.array] = None) -> Tuple[dict, dict, dict]:
        pass

    def get_deltas_right_vert_lists(self, x: Union[tuple, list, np.array],
                                    right_vertices: Union[tuple, list, np.array] = None) -> Tuple[list, list]:
        """

        :param right_vertices:
        :param x: input to interpolate
        :return: list of deltas and right vertices
        """
        if right_vertices is None:
            right_vertices_indices = []
            deltas = []
            for x_i, one_d_grid in zip(x, self.search_space.full_space):
                cur_right_index = find_right_vertex_index_for_x(x_i, one_d_grid)
                right_vertices_indices.append(cur_right_index)
                delta_c = (x_i - one_d_grid[cur_right_index - 1]) / (
                        one_d_grid[cur_right_index] - one_d_grid[
                    cur_right_index - 1])  # FIXME when we take x_i > 1 -> we get delta > grid step
                deltas.append(delta_c)
                # TODO assert denominator is equal to 1 / l_c

            return deltas, right_vertices_indices
        else:
            right_vertices_indices = []
            deltas = []
            for x_i, one_d_grid, cur_right_index in zip(x, self.search_space.full_space, right_vertices):

                right_vertices_indices.append(cur_right_index)
                delta_c = (x_i - one_d_grid[cur_right_index - 1]) / (
                        one_d_grid[cur_right_index] - one_d_grid[
                    cur_right_index - 1])  # FIXME when we take x_i > 1 -> we get delta > grid step
                deltas.append(delta_c)
                # TODO assert denominator is equal to 1 / l_c

            return deltas, right_vertices_indices

    @staticmethod
    def get_index(point_side: str):
        def right_index(index: int) -> int:
            return index

        def left_index(index: int) -> int:
            return max(index - 1, 0)

        if point_side == 'l':
            return left_index

        if point_side == 'r':
            return right_index

    @staticmethod
    def get_delta_multiplier(point_side: str):
        def direct_delta_multiplier(weight: float) -> float:
            return weight

        def indirect_delta_multiplier(weight: float) -> float:
            return 1 - weight

        if point_side == 'l':
            return indirect_delta_multiplier
        if point_side == 'r':
            return direct_delta_multiplier

    def get_weights_for_full_grid(self, x: Union[tuple, list, np.array],
                                  right_vertices: Union[tuple, list, np.array] = None) -> Tuple[dict, dict]:
        """

        :param x: input to interpolate
        :return: dict of weights for each vertex in the grid
        """
        _, weights_dict, index_mapper = self.all_weights_for_cube(x, right_vertices)
        indexed_weights = {}
        indexed_weights_single_index = {}
        for key, value in weights_dict.items():
            indexed_weights[index_mapper[key]] = value
            indexed_weights_single_index[self.search_space.get_j_from_jr_jg_rb(*index_mapper[key])] = value

        return indexed_weights, indexed_weights_single_index

    def interpolate(self, x: Union[tuple, list, np.array]) -> Union[tuple, list, np.array]:
        vertices_dict, weights_dict, _ = self.all_weights_for_cube(x)
        interpolated_x = np.array([0, 0, 0], dtype=float)
        for key in weights_dict.keys():
            interpolated_x += weights_dict[key] * vertices_dict[key]

        return interpolated_x


class TetrahedralInterpolation(Interpolation):
    def all_weights_for_cube(self, x: Union[tuple, list, np.array]) -> Tuple[dict, dict, dict]:
        """
        direct cube vertices and corresponding weights calculation
        :param x: input to interpolate
        :return: dicts with vertices' coordinates and corresponding weights
        """
        deltas, right_vertices_indices = self.get_deltas_right_vert_lists(x)
        weights_dict = {x + y + z: 0 for x in ['l', 'r'] for y in ['l', 'r'] for z in ['l', 'r']}
        delta_r, delta_g, delta_b = deltas
        if delta_r <= delta_g <= delta_b:
            weights_dict['lll'] = 1 - delta_b
            weights_dict['lrr'] = delta_g - delta_r
            weights_dict['llr'] = delta_b - delta_g
            weights_dict['rrr'] = delta_r

        elif delta_r <= delta_b <= delta_g:
            weights_dict['lll'] = 1 - delta_g
            weights_dict['lrr'] = delta_b - delta_r
            weights_dict['lrl'] = delta_g - delta_b
            weights_dict['rrr'] = delta_r

        elif delta_b <= delta_r <= delta_g:
            weights_dict['lll'] = 1 - delta_g
            weights_dict['rrl'] = delta_r - delta_b
            weights_dict['lrl'] = delta_g - delta_r
            weights_dict['rrr'] = delta_b

        elif delta_b <= delta_g <= delta_r:
            weights_dict['lll'] = 1 - delta_r
            weights_dict['rrl'] = delta_g - delta_b
            weights_dict['rll'] = delta_r - delta_g
            weights_dict['rrr'] = delta_b

        elif delta_g <= delta_b <= delta_r:
            weights_dict['lll'] = 1 - delta_r
            weights_dict['rlr'] = delta_b - delta_g
            weights_dict['rll'] = delta_r - delta_b
            weights_dict['rrr'] = delta_g

        elif delta_g <= delta_r <= delta_b:
            weights_dict['lll'] = 1 - delta_b
            weights_dict['rlr'] = delta_r - delta_g
            weights_dict['llr'] = delta_b - delta_r
            weights_dict['rrr'] = delta_g

        vertices_dict = weights_dict.copy()
        character_to_index = {}
        for key in weights_dict.keys():
            vertices_dict[key] = np.array([
                self.search_space.full_space[0][self.get_index(key[0])(right_vertices_indices[0])],
                self.search_space.full_space[1][self.get_index(key[1])(right_vertices_indices[1])],
                self.search_space.full_space[2][self.get_index(key[2])(right_vertices_indices[2])]])
            character_to_index[key] = (self.get_index(key[0])(right_vertices_indices[0]),
                                       self.get_index(key[1])(right_vertices_indices[1]),
                                       self.get_index(key[2])(right_vertices_indices[2]))

        return vertices_dict, weights_dict, character_to_index


class TrilinearInterpolation(Interpolation):

    def all_weights_for_cube(self, x: Union[tuple, list, np.array],
                             right_vertices: Union[tuple, list, np.array] = None) -> Tuple[dict, dict, dict]:
        """
        direct cube vertices and corresponding weights calculation
        :param x: input to interpolate
        :return: dicts with vertices' coordinates and corresponding weights
        """
        deltas, right_vertices_indices = self.get_deltas_right_vert_lists(x, right_vertices)
        weights_dict = {x + y + z: 0 for x in ['l', 'r'] for y in ['l', 'r'] for z in ['l', 'r']}

        vertices_dict = weights_dict.copy()
        character_to_index = {}
        for key in weights_dict.keys():
            weights_dict[key] = self.get_delta_multiplier(key[0])(deltas[0]) * \
                                self.get_delta_multiplier(key[1])(deltas[1]) * \
                                self.get_delta_multiplier(key[2])(deltas[2])

            vertices_dict[key] = np.array([
                self.search_space.full_space[0][self.get_index(key[0])(right_vertices_indices[0])],
                self.search_space.full_space[1][self.get_index(key[1])(right_vertices_indices[1])],
                self.search_space.full_space[2][self.get_index(key[2])(right_vertices_indices[2])]])
            character_to_index[key] = (self.get_index(key[0])(right_vertices_indices[0]),
                                       self.get_index(key[1])(right_vertices_indices[1]),
                                       self.get_index(key[2])(right_vertices_indices[2]))

        return vertices_dict, weights_dict, character_to_index

    @timing
    def interpolate_with_matrices(self, x: Union[tuple, list, np.array]) -> Union[tuple, list, np.array]:
        """
        p = Q1^T * B_1 * P
        P - cube points
        Q1^T - deltas
        B_1 - hardcoded

        :param x: input to interpolate
        :return: interpolated value
        """
        deltas, right_vertices_indices = self.get_deltas_right_vert_lists(x)
        vertices_dict = {x + y + z: 0 for x in ['l', 'r'] for y in ['l', 'r'] for z in ['l', 'r']}

        for key in vertices_dict.keys():
            vertices_dict[key] = np.array([
                self.search_space.full_space[0][self.get_index(key[0])(right_vertices_indices[0])],
                self.search_space.full_space[1][self.get_index(key[1])(right_vertices_indices[1])],
                self.search_space.full_space[2][self.get_index(key[2])(right_vertices_indices[2])]])
        matrix_p = np.array([vertices_dict['lll'],
                             vertices_dict['llr'],
                             vertices_dict['lrl'],
                             vertices_dict['lrr'],

                             vertices_dict['rll'],
                             vertices_dict['rlr'],
                             vertices_dict['rrl'],
                             vertices_dict['rrr']])

        matrix_q1 = np.array([1, deltas[0], deltas[1], deltas[2],
                              deltas[0] * deltas[1], deltas[1] * deltas[2], deltas[2] * deltas[0],
                              deltas[0] * deltas[1] * deltas[2]])

        matrix_b1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [-1, 0, 0, 0, 1, 0, 0, 0],
                              [-1, 0, 1, 0, 0, 0, 0, 0],
                              [-1, 1, 0, 0, 0, 0, 0, 0],

                              [1, 0, -1, 0, -1, 0, 1, 0],
                              [1, -1, -1, 1, 0, 0, 0, 0],
                              [1, -1, 0, 0, -1, 1, 0, 0],
                              [-1, 1, 1, -1, 1, -1, -1, 1]])

        # p = Q1 ^ T * B_1 * P

        return matrix_q1.T.dot(matrix_b1).dot(matrix_p)


if __name__ == '__main__':
    import sys

    sys.path.append('/media/nchudnov/extra_sp/stu/code')

    from colorimetry import colorimetry

    rgb_space = UniformSearchSpace(5, 5, 5)
    trilinear_int = TrilinearInterpolation(rgb_space)
    tetr_int = TetrahedralInterpolation(rgb_space)
    # trilinear_int.search_space.transform_space(print)
    # exit()
    vec = [0.121, 0.121, 0.121]
    print(f'tri_res = {trilinear_int.interpolate(vec)}')
    print(f'tetr_res = {tetr_int.interpolate(vec)}')

    # print(trilinear_int.get_matrices_for_interpolation([0.125, 0.125, 0.125]))
    # print(trilinear_int.get_matrices_for_interpolation([0.5, 0.5, 0.5]))
    # print(trilinear_int.interpolate([0.5, 0.5, 0.5]))
    # iterations = 10
    # start = time.time()
    # for i in range(iterations):
    #     trilinear_int.interpolate_with_matrices([0.2, 0.25, 0.85])
    # end = time.time()
    # print(f'matrix ellapsed = {(end - start) / iterations}')
    # iterations = 100
    # start = time.time()
    # for i in range(iterations):
    #     trilinear_int.interpolate([0.2, 0.25, 0.85])
    # end = time.time()
    # print(f'NO matrix ellapsed = {(end - start) / iterations}')
    # print(trilinear_int.get_matrices_for_interpolation([0.125, 0.125, 0.125]))
