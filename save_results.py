import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Union, NamedTuple

# from example import dir_for_save

class Data_for_graphs(NamedTuple):
    model_name_short: str
    model_name_full: str
    descr_param: str
    diff_train: np.array
    diff_test: np.array


macbeth_rgb = np.array([(115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67), (133, 128, 177),
                       (103, 189, 170), (214, 126, 44), (80, 91, 166), (193, 90, 99), (94, 60, 108),
                       (157, 188, 64), (224, 163, 46), (56, 61, 150), (70, 148, 73), (175, 54, 60),
                       (231, 199, 31), (187, 86, 149), (8, 133, 161), (243, 243, 242), (200, 200, 200),
                       (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52)])

def show_time(time_start, time_stop, title):
    hours, rem = divmod(time_stop - time_start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(title)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def get_txt_info(np_array, round_value = 2):
    df = pd.DataFrame()
    df['color_diff'] = np_array
    desc = df.describe()
    desc = desc.drop('count')
    desc = desc.drop('std')
    desc = desc.drop('25%')
    desc = desc.drop('75%')
    # desc = desc.rename({'mean': 'L1'})
    desc = desc.rename({'50%': 'med'})
    # desc.loc['L2'] = np.linalg.norm(np_array, 2) / len(np_array) ** 0.5
    desc.loc['90%'] = np.quantile(np_array, 0.9)
    desc.loc['95%'] = np.quantile(np_array, 0.95)
    desc = desc.reindex(['mean', 'min', 'med', '90%', '95%', 'max'])
    desc = desc.round(round_value)
    data1=[i for i in desc.index]
    if round_value == 2:
        data2=["{:.2f}".format(i) for i in desc.color_diff]
    elif round_value == 4:
        data2 = ["{:.1f}".format(i * 1000) + '     e-3' for i in desc.color_diff]
    # data2 = [str(round(i, round_value)) for i in desc.color_diff]
    text = ('\n'.join([ a +': '+ b for a,b in zip(data1,data2)]))
    return text

def draw_graph(filename : str, sss : str, reflectances_train : str, reflectances_test : str,
               white_illuminant_name : Union[str, list], color_difference_model : str,
               data : list, save_flag = False, dir_for_save = "results/"):
    hist_width = 0
    for model in data:
        hist_width = max(hist_width, model.diff_train.max(), model.diff_test.max())

    hist_height = 0
    bins = np.arange(0, hist_width, hist_width / 40)
    for model in data:
        bins = np.arange(0, hist_width, hist_width / 40)
        hist1 = np.histogram(model.diff_train, bins)
        hist2 = np.histogram(model.diff_test, bins)
        hist_height = max(hist_height, (hist1[0] / hist1[0].sum()).max(), (hist2[0] / hist2[0].sum()).max())
    # bins = np.arange(0, max_diff_array_test, max(max_diff_array_train, max_diff_array_test) / 40)

    for model in data:
        fig, axs = plt.subplots(2, 1)
        fig.tight_layout(pad=1.5)
        fig.set_size_inches(6, 9)
        round_value = 2
        if isinstance(white_illuminant_name, str):
            fig.suptitle('SSS - ' + sss + ', illum - ' + white_illuminant_name + '\n' + model.model_name_full + ', ' +
                         model.descr_param, fontsize=10, fontweight='bold', y=1)
        elif isinstance(white_illuminant_name, list):
            round_value = 4
            fig.suptitle('SSS - ' + sss + ', illum 1 - ' + white_illuminant_name[0] +
                         ' illum 2 - ' + white_illuminant_name[1] + '\n' + model.model_name_full + ', ' +
                         model.descr_param,
                         fontsize=10, fontweight='bold', y=1)

        # fig.text(0.54, 0.98, descr, fontsize=10, ha='center', va='top')
        ax = axs[0]
        ax.text(0.95, 0.92, get_txt_info(model.diff_train, round_value), fontsize=10, ha='right', va='top', transform=ax.transAxes,
                bbox={'facecolor': 'whitesmoke', 'alpha': 0.5, 'pad': 10})
        ax.set_xlabel('Color difference ' + color_difference_model)
        ax.set_ylabel("Probability")
        ax.set_title('Reflectances: ' + reflectances_train)
        ax.grid(color='whitesmoke', linestyle='--', linewidth=1, alpha = 0.5)

        # max_diff_array_train = np.array(model.diff_train).max()
        # bins = np.arange(0, max_diff_array_train, max_diff_array_train / 40)
        # bins = np.arange(0, hist_width, hist_width / 40)
        hist = np.histogram(model.diff_train, bins)
        # hist_height = (hist[0] / hist[0].sum()).max()
        ax.bar(hist[1][:-1], hist[0] / hist[0].sum(), hist[1][1] - hist[1][0], facecolor='green')
        ax.set_ylim([0, hist_height + 0.02])
        ax.grid(color='whitesmoke', linestyle='--', linewidth=1, alpha=0.5)

        ax = axs[1]
        ax.text(0.95, 0.92, get_txt_info(model.diff_test, round_value), fontsize=10, ha='right', va='top', transform=ax.transAxes,
                bbox={'facecolor': 'whitesmoke', 'alpha': 0.5, 'pad': 10})
        ax.set_xlabel('Color difference ' + color_difference_model)
        ax.set_ylabel("Probability")
        ax.set_title('Reflectances: ' + reflectances_test)
        ax.grid(color='whitesmoke', linestyle='--', linewidth=1, alpha=0.5)

        # max_diff_array_test = np.array(model.diff_test).max()
        # bins = np.arange(0, max_diff_array_test, max_diff_array_test / 40)
        # bins = np.arange(0, hist_width, hist_width / 40)
        hist = np.histogram(model.diff_test, bins)
        # hist_height = (hist[0] / hist[0].sum()).max()
        ax.bar(hist[1][:-1], hist[0] / hist[0].sum(), hist[1][1] - hist[1][0], facecolor='green')
        ax.set_ylim([0, hist_height + 0.02])
        ax.grid(color='whitesmoke', linestyle='--', linewidth=1, alpha=0.5)

        fig.tight_layout()
        plt.show()
        if (save_flag):
            file_name = dir_for_save + filename + '_' + model.model_name_short + '_' + model.descr_param + '.png'
            print("Graphs for " + model.model_name_full + "successfully generated and saved into: " + file_name)
            fig.savefig(file_name, dpi=300, facecolor='white', bbox_inches='tight')


def draw_graph_for_gamut_mapping(filename, settings, color_difference_model, diff_array_train, diff_array_test, save_flag=False, dir_for_save="results/"):
    max_diff_array_train = np.array(diff_array_train).max()
    max_diff_array_test = np.array(diff_array_test).max()
    width = max(max_diff_array_train, max_diff_array_test)
    # bins = np.arange(0, max_diff_array_test, max(max_diff_array_train, max_diff_array_test) / 40)

    fig, axs = plt.subplots(2, 1)
    fig.tight_layout(pad=1.5)
    # fig.set_size_inches(8, 12)
    fig.set_size_inches(6, 9)
    fig.suptitle('input gamut - ' + settings['input_gamut'] + ', output gamut - ' + settings['output_gamut'],
                 fontsize=10, fontweight='bold', y=1)
    # fig.text(0.54, 0.98, descr, fontsize=10, ha='center', va='top')
    ax = axs[0]

    ax.text(0.95, 0.92, get_txt_info(diff_array_train), fontsize=10, ha='right', va='top', transform=ax.transAxes,
            bbox={'facecolor': 'whitesmoke', 'alpha': 0.5, 'pad': 10})
    ax.set_xlabel('Color difference ' + color_difference_model)
    ax.set_ylabel("Probability")
    ax.set_title('Grid_frequency_train: ' + str(settings['grid_frequency_train']))
    ax.grid(color='whitesmoke', linestyle='--', linewidth=1, alpha=0.5)

    # max_diff_array_train = np.array(diff_array_train).max()
    # bins = np.arange(0, max_diff_array_train, max_diff_array_train / 40)
    bins = np.arange(0, width, width / 40)
    hist = np.histogram(diff_array_train, bins)
    hist_height = (hist[0] / hist[0].sum()).max()
    ax.bar(hist[1][:-1], hist[0] / hist[0].sum(), hist[1][1] - hist[1][0], facecolor='green')
    ax.set_ylim([0, hist_height + 0.02])
    ax.grid(color='whitesmoke', linestyle='--', linewidth=1, alpha=0.5)

    ax = axs[1]
    ax.text(0.95, 0.92, get_txt_info(diff_array_test), fontsize=10, ha='right', va='top', transform=ax.transAxes,
            bbox={'facecolor': 'whitesmoke', 'alpha': 0.5, 'pad': 10})
    ax.set_xlabel('Color difference ' + color_difference_model)
    ax.set_ylabel("Probability")
    ax.set_title('Grid_frequency_test: ' + str(settings['grid_frequency_test']))
    ax.grid(color='whitesmoke', linestyle='--', linewidth=1, alpha=0.5)

    # max_diff_array_test = np.array(diff_array_test).max()
    # bins = np.arange(0, max_diff_array_test, max_diff_array_test / 40)
    bins = np.arange(0, width, width / 40)
    hist = np.histogram(diff_array_test, bins)
    # hist_height = (hist[0] / hist[0].sum()).max()
    ax.bar(hist[1][:-1], hist[0] / hist[0].sum(), hist[1][1] - hist[1][0], facecolor='green')
    ax.set_ylim([0, hist_height + 0.02])
    ax.grid(color='whitesmoke', linestyle='--', linewidth=1, alpha=0.5)

    fig.tight_layout()
    plt.show()
    if (save_flag):
        file_name = dir_for_save + filename + '.png'
        print("\nGraphs successfully generated and saved into: " + file_name)
        fig.savefig(file_name, dpi=300, facecolor='white', bbox_inches='tight')

def save_data(filename, data, a_opt, b_opt, X, Y, Y_res, save_flag = False, dir_for_save = "results/"):
    if save_flag == False:
        return

    df_results = pd.DataFrame(data.items(), columns=['description', 'value'])
    df_results[" "] = ""

    df_a_opt = pd.DataFrame()
    df_a_opt['a_opt_0'] = a_opt[0]
    df_a_opt['a_opt_1'] = a_opt[1]
    df_a_opt['a_opt_2'] = a_opt[2]
    df_a_opt[" "] = ""

    df_b_opt = pd.DataFrame()
    df_b_opt['b_opt_0'] = b_opt[:, 0]
    df_b_opt['b_opt_1'] = b_opt[:, 1]
    df_b_opt['b_opt_2'] = b_opt[:, 2]
    df_b_opt[" "] = ""

    dfX = pd.DataFrame()
    dfX['X0'] = X[:, 0]
    dfX['X1'] = X[:, 1]
    dfX['X2'] = X[:, 2]
    dfX[" "] = ""

    dfY = pd.DataFrame()
    dfY['Y0'] = Y[:, 0]
    dfY['Y1'] = Y[:, 1]
    dfY['Y2'] = Y[:, 2]
    dfY[" "] = ""

    dfY['Y0^'] = Y_res[:, 0]
    dfY['Y1^'] = Y_res[:, 1]
    dfY['Y2^'] = Y_res[:, 2]

    df_final = pd.concat([df_results, df_a_opt, df_b_opt, dfX, dfY], axis=1)
    file_name = dir_for_save + filename + '.csv'
    df_final.to_csv(file_name)
    print("\nData successfully saved into: " + file_name)

def create_lines_for_plot(x0, x1):
    x_lines = []
    y_lines = []
    z_lines = []

    for i in range(x0.shape[0]):
        x_lines.append(x0[i, 0])
        x_lines.append(x1[i, 0])
        x_lines.append(None)

        y_lines.append(x0[i, 1])
        y_lines.append(x1[i, 1])
        y_lines.append(None)

        z_lines.append(x0[i, 2])
        z_lines.append(x1[i, 2])
        z_lines.append(None)

    return x_lines, y_lines, z_lines

def plot_3d_graph(filename, a_opt, b_opt,
                  X_train, Y_train, Y_train_hat,
                  X_test, Y_test, Y_test_hat,
                  save_flag=False, dir_for_save ="results/"):
    a0, a1, a2 = np.meshgrid(a_opt[0], a_opt[1], a_opt[2], indexing='ij')
    a_opt_points = np.stack((a0, a1, a2), axis=3).reshape(len(a0) * len(a1) * len(a2), 3)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=a_opt_points[:, 0],
                               y=a_opt_points[:, 1],
                               z=a_opt_points[:, 2],
                               mode='markers',
                               name="a_opt",
                               visible='legendonly',
                               marker=dict(size=5,
                                           color='olive',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    fig.add_trace(go.Scatter3d(x=b_opt[:, 0],
                               y=b_opt[:, 1],
                               z=b_opt[:, 2],
                               mode='markers',
                               name="b_opt",
                               marker=dict(size=5,
                                           color='maroon',
                                           colorscale='Viridis',
                                           opacity=0.5)
                                   ))

    fig.add_trace(go.Scatter3d(x=X_train[:, 0],
                               y=X_train[:, 1],
                               z=X_train[:, 2],
                               mode='markers',
                               name="X train",
                               visible='legendonly',
                               marker=dict(size=2,
                                           color='slateblue',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    fig.add_trace(go.Scatter3d(x=Y_train[:, 0],
                               y=Y_train[:, 1],
                               z=Y_train[:, 2],
                               mode='markers',
                               name="Y train",
                               visible='legendonly',
                               marker=dict(size=2,
                                           color='blue',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    fig.add_trace(go.Scatter3d(x=Y_train_hat[:, 0],
                               y=Y_train_hat[:, 1],
                               z=Y_train_hat[:, 2],
                               mode='markers',
                               name="Y^ train",
                               visible='legendonly',
                               marker=dict(size=2,
                                           color='darkblue',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    train_x_lines, train_y_lines, train_z_lines = create_lines_for_plot(Y_train, Y_train_hat)

    fig.add_trace(go.Scatter3d(x=train_x_lines,
                               y=train_y_lines,
                               z=train_z_lines,
                               mode='lines',
                               name="Y-Y^ train",
                               visible='legendonly',
                               line=dict(
                                   color='midnightblue',
                                   width=2
                               )
                               ))

    train_x_lines_xy, train_y_lines_xy, train_z_lines_xy = create_lines_for_plot(X_train, Y_train)

    fig.add_trace(go.Scatter3d(x=train_x_lines_xy,
                               y=train_y_lines_xy,
                               z=train_z_lines_xy,
                               mode='lines',
                               name="X-Y train",
                               visible='legendonly',
                               line=dict(
                                   color='blue',
                                   width=2
                               )
                               ))

    # -----------------------------------------------------------------------

    fig.add_trace(go.Scatter3d(x=X_test[:, 0],
                               y=X_test[:, 1],
                               z=X_test[:, 2],
                               mode='markers',
                               name="X test",
                               visible='legendonly',
                               marker=dict(size=2,
                                           color='salmon',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    fig.add_trace(go.Scatter3d(x=Y_test[:, 0],
                               y=Y_test[:, 1],
                               z=Y_test[:, 2],
                               mode='markers',
                               name="Y test",
                               visible='legendonly',
                               marker=dict(size=2,
                                           color='red',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    fig.add_trace(go.Scatter3d(x=Y_test_hat[:, 0],
                               y=Y_test_hat[:, 1],
                               z=Y_test_hat[:, 2],
                               mode='markers',
                               name="Y^ test",
                               visible='legendonly',
                               marker=dict(size=2,
                                           color='firebrick',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    test_x_lines, test_y_lines, test_z_lines = create_lines_for_plot(Y_test, Y_test_hat)

    fig.add_trace(go.Scatter3d(x=test_x_lines,
                               y=test_y_lines,
                               z=test_z_lines,
                               mode='lines',
                               name="Y-Y^ test",
                               visible='legendonly',
                               line=dict(
                                   color='darkred',
                                   width=2
                               )
                               ))

    test_x_lines_xy, test_y_lines_xy, test_z_lines_xy = create_lines_for_plot(X_test, Y_test)

    fig.add_trace(go.Scatter3d(x=test_x_lines_xy,
                               y=test_y_lines_xy,
                               z=test_z_lines_xy,
                               mode='lines',
                               name="X-Y test",
                               visible='legendonly',
                               line=dict(
                                   color='red',
                                   width=2
                               )
                               ))

    fig.update_xaxes(title_font=dict(size=18, family='Courier'))

    axis_names = ['R', 'G', 'B']

    fig.update_layout(
        autosize=False,
        width=950,
        height=800,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
        paper_bgcolor="White",
        scene=dict(
            xaxis_title=axis_names[0],
            yaxis_title=axis_names[1],
            zaxis_title=axis_names[2],
        )
    )

    fig.show()
    if save_flag:
        file_name = dir_for_save + filename
        fig.write_html(file_name + ".html")


def plot_3d_graph_prolab(filename,
                         X_train, Y_train, Y_train_hat,
                         X_test, Y_test, Y_test_hat,
                         save_flag=False, dir_for_save ="results/"):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=X_train[:, 0],
                               y=X_train[:, 1],
                               z=X_train[:, 2],
                               mode='markers',
                               name="X train",
                               visible='legendonly',
                               marker=dict(size=2,
                                           color='slateblue',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    fig.add_trace(go.Scatter3d(x=Y_train[:, 0],
                               y=Y_train[:, 1],
                               z=Y_train[:, 2],
                               mode='markers',
                               name="Y train",
                               # visible='legendonly',
                               marker=dict(size=2,
                                           color='red',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    fig.add_trace(go.Scatter3d(x=Y_train_hat[:, 0],
                               y=Y_train_hat[:, 1],
                               z=Y_train_hat[:, 2],
                               mode='markers',
                               name="Y^ train",
                               visible='legendonly',
                               marker=dict(size=2,
                                           color='darkblue',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    train_x_lines_, train_y_lines_, train_z_lines_ = create_lines_for_plot(X_train, Y_train)

    fig.add_trace(go.Scatter3d(x=train_x_lines_,
                               y=train_y_lines_,
                               z=train_z_lines_,
                               mode='lines',
                               name="X-Y train",
                               visible='legendonly',
                               line=dict(
                                   color='blue',
                                   width=2
                               )
                               ))

    train_x_lines, train_y_lines, train_z_lines = create_lines_for_plot(Y_train, Y_train_hat)

    fig.add_trace(go.Scatter3d(x=train_x_lines,
                               y=train_y_lines,
                               z=train_z_lines,
                               mode='lines',
                               name="Y-Y^ train",
                               visible='legendonly',
                               line=dict(
                                   color='midnightblue',
                                   width=2
                               )
                               ))

    # -----------------------------------------------------------------------

    fig.add_trace(go.Scatter3d(x=X_test[:, 0],
                               y=X_test[:, 1],
                               z=X_test[:, 2],
                               mode='markers',
                               name="X test",
                               visible='legendonly',
                               marker=dict(size=2,
                                           color='salmon',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    fig.add_trace(go.Scatter3d(x=Y_test[:, 0],
                               y=Y_test[:, 1],
                               z=Y_test[:, 2],
                               mode='markers',
                               name="Y test",
                               visible='legendonly',
                               marker=dict(size=2,
                                           color='red',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    fig.add_trace(go.Scatter3d(x=Y_test_hat[:, 0],
                               y=Y_test_hat[:, 1],
                               z=Y_test_hat[:, 2],
                               mode='markers',
                               name="Y^ test",
                               visible='legendonly',
                               marker=dict(size=2,
                                           color='firebrick',
                                           colorscale='Viridis',
                                           opacity=0.5)
                               ))

    test_x_lines_, test_y_lines_, test_z_lines_ = create_lines_for_plot(X_test, Y_test)

    fig.add_trace(go.Scatter3d(x=test_x_lines_,
                               y=test_y_lines_,
                               z=test_z_lines_,
                               mode='lines',
                               name="X-Y test",
                               visible='legendonly',
                               line=dict(
                                   color='red',
                                   width=2
                               )
                               ))

    test_x_lines, test_y_lines, test_z_lines = create_lines_for_plot(Y_test, Y_test_hat)

    fig.add_trace(go.Scatter3d(x=test_x_lines,
                               y=test_y_lines,
                               z=test_z_lines,
                               mode='lines',
                               name="Y-Y^ test",
                               visible='legendonly',
                               line=dict(
                                   color='darkred',
                                   width=2
                               )
                               ))
    #
    # fig.add_trace(go.Mesh3d(x=Y_train[:, 0],
    #                         y=Y_train[:, 1],
    #                         z=Y_train[:, 2],
    #                         opacity=0.5,
    #                         name="Y train surface",
    #                         color='red'
    #                         ))
    #
    # fig.add_trace(go.Mesh3d(x=Y_test[:, 0],
    #                         y=Y_test[:, 1],
    #                         z=Y_test[:, 2],
    #                         opacity=0.5,
    #                         name="Y test surface",
    #                         color='blue'
    #                         ))

    fig.update_xaxes(title_font=dict(size=18, family='Courier'))

    axis_names = ['L', 'a', 'b']

    fig.update_layout(
        autosize=False,
        width=950,
        height=800,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
        paper_bgcolor="White",
        scene=dict(
            xaxis_title=axis_names[0],
            yaxis_title=axis_names[1],
            zaxis_title=axis_names[2],
        )
    )

    fig.show()
    if save_flag:
        file_name = dir_for_save + filename
        fig.write_html(file_name + "_prolab.html")