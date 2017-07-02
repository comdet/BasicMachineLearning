import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import json
import matplotlib


def plot_training_group(plt,X, indices, color, edgecolor):
    #indices = indices[1]
    plot_x = X[indices, 0]
    plot_y = X[indices, 1]
    plt.plot(plot_x, plot_y, 'o',
                         markersize=15.0,
                         color=color,
                         markeredgecolor=edgecolor,
                         markeredgewidth=2.0,
                         zorder=10)

def decision_colorbar():
    negative_r = 1.000
    negative_g = 0.859
    negative_b = 0.859

    positive_r = 0.941
    positive_g = 1.000
    positive_b = 0.839

    decision_c = 1.000

    dec_delta = 0.05
    dec_boundary = 0.5
    dec_boundary_low = dec_boundary - dec_delta
    dec_boundary_high = dec_boundary + dec_delta

    c_dict = {'red': ((0.000, negative_r, negative_r),
                      (dec_boundary_low, negative_r, negative_r),
                      (dec_boundary, decision_c, decision_c),
                      (dec_boundary_high, positive_r, positive_r),
                      (1.0, positive_r, positive_r)),
              'green': ((0.00, negative_g, negative_g),
                        (dec_boundary_low, negative_g, negative_g),
                        (dec_boundary, decision_c, decision_c),
                        (dec_boundary_high, positive_g, positive_g),
                        (1.00, positive_g, positive_g)),
              'blue': ((0.0, negative_b, negative_b),
                       (dec_boundary_low, negative_b, negative_b),
                       (dec_boundary, decision_c, decision_c),
                       (dec_boundary_high, positive_b, positive_b),
                       (1.0, positive_b, positive_b))}

    colorbar = LinearSegmentedColormap('XOR Green Red', c_dict)
    return colorbar


def plot_data_group(ax, X, Z, marker, markersize):

    plot_x = X[:, 0]
    plot_y = X[:, 1]

    colorbar = decision_colorbar()

    ax.scatter(plot_x, plot_y, marker='o', c=Z, linewidths=0,
               cmap=colorbar, zorder=-1)

def plot_logic(plt,training_data,target_data):
    style = json.load(open("538.json"))
    matplotlib.rcParams.update(style)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Logic Table")

    # Training colors
    red = '#FF6B6B'
    green = '#AFE650'
    lightred = '#FFDBDB'
    lightgreen = '#F0FFD6'

    # # Plot Training 0's
    indices = np.where(np.array(target_data) < 0.5)
    plot_training_group(plt,training_data[:,:-1], indices, red, lightred)

    # # Plot Training 1's
    indices = np.where(np.array(target_data) > 0.5)
    plot_training_group(plt,training_data[:,:-1], indices, green, lightgreen)

    # splines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')

    # Axis
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([-0.1, 1.1])

    axis_x = [-0.1, 1.1]
    axis_y = [0, 0]
    ax.plot(axis_x, axis_y, color='#000000', linewidth=1.0)
    axis_x = [0, 0]
    axis_y = [-0.1, 1.1]
    ax.plot(axis_x, axis_y, color='#000000', linewidth=1.0)

    plt.gca().set_aspect('equal', adjustable='box')
    return plt
def plot_space(plt,w):
    x = np.linspace(-0.1, 1.1)
    bias = np.ones([len(x)])
    for i in range(len(x)):
        x_ = np.zeros([len(x)])
        x_.fill(x[i])
        y = np.linspace(-0.1, 1.1)
        inp = np.dstack([x_,y,bias])
        inp = np.reshape(inp,[len(x),3])
        z = np.matmul(inp,w)
        indices = np.where(np.array(z) > 0)
        plt.plot(x_[indices], y[indices], 'o',
                             markersize=1.0,
                             color='#99e05c')
        indices = np.where(np.array(z) <= 0)
        plt.plot(x_[indices], y[indices], 'o',
                             markersize=1.0,
                             color='#db7559')
    return plt