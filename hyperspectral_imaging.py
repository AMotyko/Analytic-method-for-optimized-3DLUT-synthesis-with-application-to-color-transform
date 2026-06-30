from data_generator import DataGenerator
from data_generator import DataGeneratorGamut
from typing import Tuple
from dataset import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import colour
import h5py
import cv2


class ImageProcessing(object):
    def __init__(self):
        self.case = 'RGB2XYZ'
        self.data_generator_obj = DataGenerator
        self.data_generator_gamut_obj = DataGeneratorGamut
        # self.bounds = np.array([[[1, 1, 0], [0, 1, 1]],
        #                         [[0, 0, 1], [1, 0, 0]],
        #                         [[1, 1, 0], [0, 0, 1]],
        #                         [[0, 0, 1], [0, 1, 1]],
        #                         [[1, 1, 0], [1, 0, 0]],
        #                         [[0, 1, 1], [1, 0, 0]]]) # array with left and right bounds of all strips
        self.bounds = np.array([[[0, 0, 0], [0.99, 0.99, 0.99]],
                                [[0, 1, 0], [1, 0, 1]],
                                [[0, 0.9, 0.9], [0.9, 0, 0.9]],
                                [[1, 1, 0], [0.5, 0.5, 1]],
                                [[0.5, 0, 0], [0, 0, 1]],
                                [[0.5, 0.1, 0.1], [0.6, 0.6, 0.4]],
                                [[0.1, 0.3, 0.6], [0.9, 0.8, 0.4]],
                                [[0.2, 0.2, 0.2], [0, 0.5, 0.8]]]) # array with left and right bounds of all strips

        self.rows = 7 # height of strip
        self.cols = 100 # width of strip
        self.spread = 2 # spread between different strips

    def initialize_spectral_data(self, reflectances, cmf, sss, white_illuminant, white_illuminant_name):
        self.data_generator_obj = DataGenerator()
        self.data_generator_obj.set_spectral_data(reflectances,
                                                  cmf[0],
                                                  sss[0],
                                                  white_illuminant,
                                                  white_illuminant_name,
                                                  wl_step=2,
                                                  print_info=False)

    def initialize_gamut_generator(self, data_generator_gamut_obj):
        self.data_generator_gamut_obj = data_generator_gamut_obj

    def set_case(self, val):
        self.case = val

    def get_tristimulus_values(self) -> Tuple[np.array, np.array]:
        """ :returns: arrays of tristimulus values for camera and GT (X and Y)  """
        return self.data_generator_obj.get_data()

    def prepare_image_XYZ(self, image_input):
        image_output = np.zeros(image_input.shape)
        image_ca = np.zeros(image_input.shape)
        for i in range(len(image_input)):
            image_ca[i] = self.data_generator_obj.M_CAT.dot(image_input[i])
            image_output[i] = colour.XYZ_to_sRGB(image_ca[i])

        return image_output, image_ca

    def calc_contrast_Rosenf(self, roi_prolab):
        vec = [1, 2, 1]
        Gx = 0
        Gy = 0
        for k in [-1, 0, 1]:
            Gx += np.linalg.norm(roi_prolab[1, k] - roi_prolab[-1, k]) * vec[k + 1]
            Gy += np.linalg.norm(roi_prolab[k, 1] - roi_prolab[k, -1]) * vec[k + 1]
        return np.sqrt(Gx ** 2 + Gy ** 2)

    def find_color_contrast_Rosenf(self, img_rgb, gamut_type = 'output'):
        img_prolab = self.data_generator_gamut_obj.rgb_to_prolab(img_rgb, gamut_type = gamut_type)
        img_prolab = img_prolab.reshape((self.rows, self.cols, 3))
        img_contrast = np.zeros(img_prolab.shape[:-1])
        for i in range(1, img_prolab.shape[0] - 1):
            for j in range(1, img_prolab.shape[1] - 1):
                img_contrast[i, j] = self.calc_contrast_Rosenf(img_prolab[i - 1:i + 2, j - 1:j + 2])
        return img_contrast


class GradientImage(ImageProcessing):
    def calculate_gradient(self, bounds, count):
        x = np.array([0, 1])
        x_new = np.linspace(0, 1, count)
        colors_rgb = np.zeros((len(x_new), 3))
        colors_rgb[:, 0] = colour.LinearInterpolator(x, bounds[:, 0])(x_new)
        colors_rgb[:, 1] = colour.LinearInterpolator(x, bounds[:, 1])(x_new)
        colors_rgb[:, 2] = colour.LinearInterpolator(x, bounds[:, 2])(x_new)

        return colors_rgb

    def initialize(self, path):
        strips = np.zeros((len(self.bounds) * self.cols, 3))
        M_CAT_inv = np.linalg.inv( self.data_generator_obj.M_CAT)
        for i, b in enumerate(self.bounds):
            rgb_vals = self.calculate_gradient(b, self.cols)
            xyz_vals = []
            for rgb_val in rgb_vals:
                xyz_vals.append(M_CAT_inv.dot(colour.sRGB_to_XYZ(rgb_val)))
            strips[i * self.cols : (i + 1) * self.cols] = np.array(xyz_vals)

        reflectances = self.data_generator_obj.reconstruct_reflectances(self.data_generator_obj.CMFs, strips)

        return reflectances, None

    def visualize(self, data, filename):
        resize_factor = 5
        data_prolab = np.zeros((len(data), data['GT'].shape[0]))
        img = np.ones((self.rows * self.bounds.shape[0] * len(data) + self.spread * (self.bounds.shape[0] - 1), self.cols, 3)) * 255
        img_smoothness = np.zeros(img.shape[:-1])

        for i in range(self.bounds.shape[0]):
            for j, key in enumerate(data):
                """ colored strips calculation """
                prepared_image, xyz_ca = self.prepare_image_XYZ(data[key][i * self.cols : (i + 1) * self.cols])
                prepared_image = np.repeat(prepared_image[np.newaxis, :, :], self.rows, axis=0)
                r_pos = i * self.rows * len(data) + i * self.spread + j * self.rows
                img[r_pos : r_pos + self.rows, 0 : self.cols] = prepared_image * 255

                """ stripes with smoothness calculation """
                prepared_smoothness_image = np.zeros(xyz_ca.shape[0])
                for k in range(len(prepared_smoothness_image) - 1):
                    prepared_smoothness_image[k] = self.data_generator_obj.calc_color_difference_Prolab(xyz_ca[k],
                                                                                                xyz_ca[k + 1])
                prepared_smoothness_image = np.repeat(prepared_smoothness_image[np.newaxis, :], self.rows, axis=0)
                img_smoothness[r_pos: r_pos + self.rows, 0: self.cols] = prepared_smoothness_image

        _, xyz_ca_gt = self.prepare_image_XYZ(data['GT'])

        def concat_images(imga, imgb):
            """
            Combines two color image ndarrays side-by-side.
            """
            ha, wa = imga.shape[:2]
            hb, wb = imgb.shape[:2]
            max_height = np.max([ha, hb])
            total_width = wa + wb
            new_img = np.zeros(shape=(max_height, total_width, 3))
            new_img[:ha, :wa] = imga
            new_img[:hb, wa:wa + wb] = imgb
            return new_img

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        fontColor = (0, 0, 0)
        thickness = 1
        lineType = 2

        img_resized = cv2.resize(img, (img.shape[1] * resize_factor, img.shape[0] * resize_factor), interpolation = cv2.INTER_NEAREST)
        line_width = 60
        number_line_img = 255 * np.ones((img_resized.shape[0], line_width, 3), np.uint8)
        img_resized = concat_images(number_line_img, img_resized)
        for i in range(self.bounds.shape[0]):
            for j, key in enumerate(data):
                r_pos = resize_factor * (i * self.rows * len(data) + i * self.spread + (j + 1) * self.rows)
                cv2.putText(img_resized, key,
                            (5, r_pos - 10),
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                cv2.line(img_resized, (0, r_pos), (line_width, r_pos), (0, 0, 0), thickness=thickness)

        # for j in range(data.shape[0]):
        #     prepared_image, xyz_ca = self.prepare_image_XYZ(data[j])
        #     image_prolab_diff = np.zeros(xyz_ca_gt.shape[0])
        #     for i in range(len(xyz_ca_gt)):
        #         image_prolab_diff[i] = self.data_generator_obj.calc_color_difference_Prolab(xyz_ca_gt[i],
        #                                                                                     xyz_ca[i])
        #     data_prolab[j-1] = image_prolab_diff

        for j, key in enumerate(data):
            prepared_image, xyz_ca = self.prepare_image_XYZ(data[key])
            image_prolab_diff = np.zeros(xyz_ca_gt.shape[0])
            for i in range(len(xyz_ca_gt)):
                image_prolab_diff[i] = self.data_generator_obj.calc_color_difference_Prolab(xyz_ca_gt[i],
                                                                                            xyz_ca[i])
            data_prolab[j] = image_prolab_diff

        img_prolab = np.zeros((self.rows * self.bounds.shape[0] * data_prolab.shape[0] + self.spread * (self.bounds.shape[0] - 1), self.cols))
        for i in range(self.bounds.shape[0]):
            for j in range(data_prolab.shape[0]):
                prepared_image = data_prolab[j][i * self.cols : (i + 1) * self.cols]
                prepared_image = np.repeat(prepared_image[np.newaxis, :], self.rows, axis=0)
                r_pos = i * self.rows * data_prolab.shape[0] + i * self.spread + j * self.rows
                img_prolab[r_pos : r_pos + self.rows, 0 : self.cols] = prepared_image

        img[:, :, [0, 2]] = img[:, :, [2, 0]]

        heatmap_prolab = sns.heatmap(img_prolab, xticklabels=False, yticklabels=False, vmin=0, vmax=5, cmap="gray")
        fig = heatmap_prolab.get_figure()
        plt.show()
        fig.savefig('results/hs_images/' + 'gradient_visualization_heatmap_difference.png')  #TODO create folder with images automatically

        fig.clf()
        heatmap_smoothness = sns.heatmap(img_smoothness, xticklabels=False, yticklabels=False, vmin=0, vmax=2, cmap="gray")
        fig = heatmap_smoothness.get_figure()
        plt.show()
        fig.savefig('results/hs_images/' + 'gradient_visualization_heatmap_smoothness.png')  #TODO create folder with images automatically

        cv2.imwrite('results/hs_images/' + 'gradient_visualization.png', img_resized)
        print('\nImages successfully saved into: ')
        print('results/hs_images/' + 'gradient_visualization.png')
        print('results/hs_images/' + 'gradient_visualization_heatmap_difference.png')
        print('results/hs_images/' + 'gradient_visualization_heatmap_smoothness.png')


class HyperspectralImage(ImageProcessing):
    def initialize(self, path):
        f = h5py.File(path, 'r')
        wl = np.array(f['wavelengths'])
        if 'data' in f.keys():
            radiance = np.array(f['data']).astype(np.float64)
        elif 'radiance' in f.keys():
            radiance = np.array(f['radiance']).astype(np.float64)
        else:
            raise ValueError("Cannot find spectral data in file: '" + path)
        illum = np.array(f['illuminant']).astype(np.float64)
        self.rows = radiance.shape[0]
        self.cols = radiance.shape[1]

        """ normalize illuminant and reflectances """
        radiance /= illum
        r = np.reshape(radiance, (-1, len(wl))).T
        # illum /= (illum.mean() / 100)
        # r /= r.max()

        reflectances = pd.DataFrame(r, columns=list(range(r.shape[1])))
        reflectances.insert(loc=0, column='wl', value=wl)

        illuminant = pd.DataFrame(illum, columns=['default'])
        illuminant.insert(loc=0, column='wl', value=wl)

        return reflectances, illuminant

    def visualize(self, data, filename):
        image_gt_output, image_gt_ca = self.prepare_image_XYZ(data['GT'])

        if (self.case == 'RGB2XYZ'):
            image_ls_output, image_ls_ca = self.prepare_image_XYZ(data['LS'])
            image_transformed_output, image_transformed_ca = self.prepare_image_XYZ(data['OL'])
            image_nol_output, image_nol_ca = self.prepare_image_XYZ(data['NOL'])

        elif (self.case == 'GamutInProlab'):
            image_ls_output = data['LS']
            image_ls_ca = data['LS']
            image_transformed_output = data['OL']
            image_transformed_ca = data['OL']
            image_nol_output = data['NOL']
            image_nol_ca = data['NOL']

            contrast_map_gt_output = self.find_color_contrast_Rosenf(image_gt_output, 'output')
            contrast_map_transformed_output = self.find_color_contrast_Rosenf(image_transformed_output, 'output')

        if len(data) > 4:
            contrast_map_input = self.find_color_contrast_Rosenf(data[4], 'input')

        image_gt_output = image_gt_output.reshape((self.rows, self.cols, 3)) * 255
        image_ls_output = image_ls_output.reshape((self.rows, self.cols, 3)) * 255
        image_transformed_output = image_transformed_output.reshape((self.rows, self.cols, 3)) * 255
        image_nol_output = image_nol_output.reshape((self.rows, self.cols, 3)) * 255

        # image_gt_prolab = np.zeros(data[0].shape)
        # image_transformed_prolab = np.zeros(data[0].shape)
        image_prolab_diff = np.zeros(data['GT'].shape[0])
        image_prolab_diff_ls = np.zeros(data['GT'].shape[0])
        image_prolab_diff_nol = np.zeros(data['GT'].shape[0])
        for i in range(len(data['GT'])):
            # image_gt_prolab[i] = XYZ_to_Prolab(image_gt_ca[i])
            # image_transformed_prolab[i] = XYZ_to_Prolab(image_transformed_ca[i])
            image_prolab_diff[i] = self.data_generator_obj.calc_color_difference_Prolab(image_gt_ca[i],
                                                                                        image_transformed_ca[i])
            image_prolab_diff_ls[i] = self.data_generator_obj.calc_color_difference_Prolab(image_gt_ca[i],
                                                                                        image_ls_ca[i])
            image_prolab_diff_nol[i] = self.data_generator_obj.calc_color_difference_Prolab(image_gt_ca[i],
                                                                                        image_nol_ca[i])

        # image_gt_prolab = image_gt_prolab.reshape((self.rows, self.rows, 3))
        # image_transformed_prolab = image_transformed_prolab.reshape((self.rows, self.rows, 3))
        # np.savetxt('image_gt_prolab.csv', image_gt_prolab, delimiter=",")
        # np.savetxt('image_transformed_prolab.csv', image_transformed_prolab, delimiter=",")

        image_prolab_diff = image_prolab_diff.reshape((self.rows, self.rows))
        image_prolab_diff_ls = image_prolab_diff_ls.reshape((self.rows, self.rows))
        image_prolab_diff_nol = image_prolab_diff_nol.reshape((self.rows, self.rows))

        heatmap = sns.heatmap(image_prolab_diff, xticklabels=False, yticklabels=False, vmin=0, vmax=5, cmap="gray")
        fig = heatmap.get_figure()
        plt.show()
        fig.savefig('results/hs_images/' + filename + '_a_heatmap_ol.png')

        fig.clf()
        heatmap_ls = sns.heatmap(image_prolab_diff_ls, xticklabels=False, yticklabels=False, vmin=0, vmax=5,
                                 cmap="gray")
        fig = heatmap_ls.get_figure()
        plt.show()
        fig.savefig('results/hs_images/' + filename + '_a_heatmap_ls.png')

        fig.clf()
        heatmap_nol = sns.heatmap(image_prolab_diff_nol, xticklabels=False, yticklabels=False, vmin=0, vmax=5,
                                 cmap="gray")
        fig = heatmap_nol.get_figure()
        plt.show()
        fig.savefig('results/hs_images/' + filename + '_a_heatmap_nol.png')

        image_gt_output[:, :, [0, 2]] = image_gt_output[:, :, [2, 0]]
        image_ls_output[:, :, [0, 2]] = image_ls_output[:, :, [2, 0]]
        image_transformed_output[:, :, [0, 2]] = image_transformed_output[:, :, [2, 0]]
        image_nol_output[:, :, [0, 2]] = image_nol_output[:, :, [2, 0]]

        cv2.imwrite('results/hs_images/' + filename + '_gt.png', image_gt_output)
        cv2.imwrite('results/hs_images/' + filename + '_ls.png', image_ls_output)
        cv2.imwrite('results/hs_images/' + filename + '_ol.png', image_transformed_output)
        cv2.imwrite('results/hs_images/' + filename + '_nol.png', image_nol_output)

        print('\nImages successfully saved into: ')
        print('results/hs_images/' + filename + '_gt.png')
        print('results/hs_images/' + filename + '_ls.png')
        print('results/hs_images/' + filename + '_nol.png')
        print('results/hs_images/' + filename + '_ol.png')
        print('results/hs_images/' + filename + '_a_heatmap_ol.png')
        print('results/hs_images/' + filename + '_a_heatmap_ls.png')
        print('results/hs_images/' + filename + '_a_heatmap_nol.png')

        if (self.case == 'GamutInProlab'):
            fig.clf()
            heatmap_nol = sns.heatmap(contrast_map_gt_output, xticklabels=False, yticklabels=False, vmin=0, vmax=100,
                                      cmap="gray")
            fig = heatmap_nol.get_figure()
            plt.show()
            fig.savefig('results/hs_images/' + filename + '_a_contrast_map_gt_output.png')
            print('results/hs_images/' + filename + 'contrast_map_gt_output.png')

            fig.clf()
            heatmap_nol = sns.heatmap(contrast_map_transformed_output, xticklabels=False, yticklabels=False, vmin=0, vmax=100,
                                      cmap="gray")
            fig = heatmap_nol.get_figure()
            plt.show()
            fig.savefig('results/hs_images/' + filename + '_a_contrast_map_transformed_output.png')
            print('results/hs_images/' + filename + 'contrast_map_transformed_output.png')

            if len(data) > 4:
                fig.clf()
                heatmap_nol = sns.heatmap(contrast_map_input, xticklabels=False, yticklabels=False, vmin=0, vmax=100,
                                          cmap="gray")
                fig = heatmap_nol.get_figure()
                plt.show()
                fig.savefig('results/hs_images/' + filename + '_a_contrast_heatmap_input.png')
                print('results/hs_images/' + filename + '_a_contrast_heatmap_input.png')

task_mappings = {
    'hyperspectral': HyperspectralImage,
    'gradients': GradientImage,
}

class TestImage(object):
    def __init__(self, case, mode):
        self.available_parameters = {
            'case': ['RGB2XYZ', 'RGB2RGB', 'GamutInProlab'],
            'mode': ['gradients', 'hyperspectral']
        }

        if mode in self.available_parameters['mode']:
            self.current_task_class = task_mappings[mode]()
        else:
            raise ValueError(
                "Parameter '" + mode + "' for class TestImage() is incorrect. Please, use one of available parameters: "
                + str(self.available_parameters['mode']))

        if case in self.available_parameters['case']:
            self.current_task_class.set_case(case)
        else:
            raise ValueError(
                "Parameter '" + case + "' for class TestImage() is incorrect. Please, use one of available parameters: "
                + str(self.available_parameters['case']))



    def initialize(self, path, cmf, sss, white_illuminant, white_illuminant_name):
        dataset = Dataset()
        reflectances = dataset.reflectances.sfu.macbeth()
        self.current_task_class.initialize_spectral_data(reflectances[0], cmf, sss, white_illuminant, white_illuminant_name)
        image_reflectances, image_illuminant = self.current_task_class.initialize(path)
        self.current_task_class.initialize_spectral_data(image_reflectances, cmf, sss, white_illuminant,
                                                         white_illuminant_name)

        return image_reflectances, image_illuminant

    def initialize_gamut_generator(self, data_generator_gamut_obj):
        self.current_task_class.initialize_gamut_generator(data_generator_gamut_obj)

    def get_shape(self):
        return np.array([self.current_task_class.rows, self.current_task_class.cols, 3])

    def get_tristimulus_values(self):
        return self.current_task_class.get_tristimulus_values()

    def visualize(self, data, filename):
        return self.current_task_class.visualize(data, filename)