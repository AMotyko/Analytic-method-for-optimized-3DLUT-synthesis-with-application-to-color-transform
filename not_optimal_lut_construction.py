import numpy as np
from uniform_search_space import Interpolation, UniformSearchSpace, TrilinearInterpolation
from data_generator import DataGenerator
from dataset import Dataset


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def closest_center(point, dict_with_points):
    tmp_centers = {index: key for index, key in enumerate(dict_with_points.keys())}
    return tmp_centers[closest_node(point, np.array(list(dict_with_points.values())))]


from uniform_search_space import plot_dict


#
#
# class NonOptimalLutConstructor(object):
#     def __init__(self, number_of_nodes_in_R, data_generator):
#         boundaries = [[0, 1], [0, 1], [0, 1]]
#         number_of_nodes_in_R = number_of_nodes_in_R
#         self.rgb_grid = data_generator.get_rgb_grid(train_grid_frequency, boundaries)
#         self.space = IrregularSearchSpace(rgb_grid_train, X_train, Y_train)
#         trilinear_int = TrilinearInterpolation(space)
#         mats = NotOptimalLUTGrid(trilinear_int, space)


def get_cube(start_point, points: dict, single2multi: dict):
    cube_vertices_indices = []
    cube_center = np.zeros(3)
    j_r, j_b, j_g = start_point

    for j_r1 in [j_r, j_r + 1]:
        for j_b1 in [j_b, j_b + 1]:
            for j_g1 in [j_g, j_g + 1]:
                cur_j = single2multi[(j_r1, j_b1, j_g1)]
                cube_vertices_indices.append(cur_j)
                cube_center += points[cur_j] / 8.0
                # cube_vertices_indices[cur_j] = points[cur_j]
    return cube_vertices_indices, cube_center

    j_r + (self.l_r + 1) * j_g + (self.l_r + 1) * (self.l_g + 1) * j_b


class IrregularSearchSpace(UniformSearchSpace):
    def __init__(self, points_of_space: np.array, x: np.array, y: np.array, single2multi: dict) -> None:
        cur_grid_size = int(len(points_of_space) ** (1 / 3)) + 1
        super().__init__(number_of_points_in_r=cur_grid_size,
                         number_of_points_in_g=cur_grid_size,
                         number_of_points_in_b=cur_grid_size, do=False)
        self.x_indices = {}
        self.y_indices = {}
        for index, item in enumerate(x):
            cur_index = closest_node(item, points_of_space)
            self.x_indices[cur_index] = item
            self.y_indices[cur_index] = y[index]

        self.all_vertices_single = {k: v for k, v in enumerate(points_of_space)}

        self.multi2single = {v: k for k, v in single2multi.items()}
        self.single2multi = single2multi
        self.get_all_cubes(cur_grid_size)
        self.cube_vertices, self.cube_centers = self.filter_cubes(cur_grid_size)

    def get_all_cubes(self, number_of_points):
        cube_index = 0
        cube_centers = {}
        cube_vertices = {}
        for i in range(number_of_points - 1):
            for j in range(number_of_points - 1):
                for k in range(number_of_points - 1):
                    cube_vertices[cube_index], cube_centers[cube_index] = get_cube((i, j, k),
                                                                                   self.all_vertices_single,
                                                                                   self.multi2single)
                    cube_index += 1
        return cube_vertices, cube_centers

    def filter_cubes(self, cur_grid_size):
        cube_vertices, cube_centers = self.get_all_cubes(cur_grid_size)

        all_keys = self.all_vertices_single.keys()
        missing = [x for x in all_keys if x not in self.y_indices.keys()]
        cube_white_indices = []
        for index, cube in cube_vertices.items():
            if len(set(cube).intersection(missing)) == 0:
                cube_white_indices.append(index)
        cube_vertices = {k: v for k, v in cube_vertices.items() if k in cube_white_indices}
        cube_centers = {k: v for k, v in cube_centers.items() if k in cube_white_indices}

        return cube_vertices, cube_centers

    def get_closest_cube_center_right_vert(self, x):
        cube_index = closest_center(x, self.cube_centers)
        tmp_list = [self.single2multi[index] for index in self.cube_vertices[cube_index]]
        return (max([x[0] for x in tmp_list]),
                max([x[1] for x in tmp_list]),
                max([x[2] for x in tmp_list]))


class NotOptimalLUTGrid(object):
    def __init__(self, interpolation: Interpolation, space: IrregularSearchSpace):
        self.interpolation = interpolation
        self.Y = space.y_indices
        self.lut_in_Y = space.y_indices
        self.space = space

    def predict(self, X: np.array) -> np.array:
        pred = []
        for key, x in enumerate(X):
            res = [0, 0, 0]
            cur_right_index = self.space.get_closest_cube_center_right_vert(x)
            # cur_right_index = None
            _, cur_interpolation_results = self.interpolation.get_weights_for_full_grid(x, cur_right_index)

            for kw, vw in cur_interpolation_results.items():
                for i in range(3):
                    try:
                        res[i] += vw * self.lut_in_Y[kw][i]
                    except KeyError:
                        print("NOT IN THE GRID. The result is incorrect")

            pred.append(res)
        return np.array(pred)


if __name__ == '__main__':
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dataset_non_opt_lut = Dataset()
    not_opt_grid_frequency = 4
    boundaries = [[0, 1], [0, 1], [0, 1]]
    save_results_flag_non_opt_lut = True

    reflectances_train_non_opt_lut = dataset_non_opt_lut.reflectances.sfu.munsell_extended()

    cmf_non_opt_lut = dataset_non_opt_lut.spectral_sensitivity.xyz_matching_fun_31()
    sss_non_opt_lut = dataset_non_opt_lut.spectral_sensitivity.sonydxc930()

    white_illuminant_non_opt_lut = dataset_non_opt_lut.illuminant.cie.std()
    white_illuminant_name_non_opt_lut = 'D65'

    """ set color difference space ( CIEDE2000, Prolab ) """
    color_difference_model_non_opt_lut = "Prolab"

    """ create an instance of the class and set the required parameters """
    data_generator_obj_non_opt_lut = DataGenerator()
    data_generator_obj_non_opt_lut.set_color_difference_model(color_difference_model_non_opt_lut)

    print("Train data initialization...")

    data_generator_obj_non_opt_lut.set_spectral_data(reflectances_train_non_opt_lut[0], cmf_non_opt_lut[0],
                                                     sss_non_opt_lut[0], white_illuminant_non_opt_lut,
                                                     white_illuminant_name_non_opt_lut,
                                                     wl_step=1,
                                                     print_info=True)

    rgb_grid_train_non_opt_lut = data_generator_obj_non_opt_lut.get_rgb_grid(not_opt_grid_frequency, boundaries)
    aldas = UniformSearchSpace(not_opt_grid_frequency, not_opt_grid_frequency, not_opt_grid_frequency)

    vertices_index = aldas.all_vertices_single.keys()
    vertices_points = aldas.all_vertices_single.values()
    vertices_index, vertices_points = zip(*sorted(zip(vertices_index, vertices_points)))
    vertices_index = list(vertices_index)
    vertices_points = np.array(list(vertices_points))
    rgb_grid_train_non_opt_lut = vertices_points
    reflectances_train_non_opt_lut = [
        data_generator_obj_non_opt_lut.reconstruct_reflectances(data_generator_obj_non_opt_lut.SSS,
                                                                rgb_grid_train_non_opt_lut),
        'reconstructed, frequency: '
        + str(not_opt_grid_frequency)]
    reflectances_delta_non_opt_lut = dataset_non_opt_lut.reflectances.monochromatic.monochromatic()

    data_generator_obj_non_opt_lut.set_spectral_data(reflectances_train_non_opt_lut[0], cmf_non_opt_lut[0],
                                                     sss_non_opt_lut[0], white_illuminant_non_opt_lut,
                                                     white_illuminant_name_non_opt_lut,
                                                     wl_step=1, print_info=True)
    X_train_for_opt_lut, Y_train_for_opt_lut = data_generator_obj_non_opt_lut.get_data()
    space = IrregularSearchSpace(rgb_grid_train_non_opt_lut, X_train_for_opt_lut, Y_train_for_opt_lut,
                                 aldas.single_to_multi_map)
    trilinear_int = TrilinearInterpolation(space)
    not_optimal_LUT = NotOptimalLUTGrid(trilinear_int, space)
    print(not_optimal_LUT.predict(np.array([[0.55, 0.55, 0.55], [2.50000000e-01, 2.50000000e-01, 2.50000000e-01]])))
