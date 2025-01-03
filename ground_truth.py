import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob
import os
def plot_with_slider_1D(data):
    cur_idx = 0

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)

    idxs = []

    for d in range(len(data)):
        plt.plot(data[d], lw=3, label='Data ' + str(d))
        idxs.append(plt.plot(cur_idx, data[d][cur_idx], '.', ms=9, label='Data ' + str(d) + ' index ' + str(cur_idx)))

    axidx = plt.axes([0.25, 0.15, 0.65, 0.03])
    idx_slider = Slider(axidx, 'Index', 0, len(data[0]) - 1, valinit=cur_idx)

    def update(val):
        cur_idx = int(idx_slider.val)
        for d in range(len(data)):
            idxs[d][0].set_xdata([cur_idx])
            idxs[d][0].set_ydata(data[d][cur_idx])

    idx_slider.on_changed(update)
    ax.legend()
    ax.set_xlabel('indices')
    plt.show()


def plot_with_slider_3D(data):
    cur_idx = 0
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.subplots_adjust(bottom=0.35)
    ax.plot3D(data[:, 0], data[:, 1], data[:, 2], 'k', lw=3)
    dot, = ax.plot3D(data[cur_idx, 0], data[cur_idx, 1], data[cur_idx, 2], 'k.', ms=9)

    axidx = plt.axes([0.25, 0.15, 0.65, 0.03])
    idx_slider = Slider(axidx, 'Index', 0, len(data) - 1, valinit=cur_idx)

    def update(val):
        cur_idx = int(idx_slider.val)
        dot.set_xdata([data[cur_idx, 0]])
        dot.set_ydata([data[cur_idx, 1]])
        dot.set_3d_properties([data[cur_idx, 2]])
        # fig.canvas.draw()

    idx_slider.on_changed(update)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


if __name__ == '__main__':
    path = '/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/xyz data/full_tasks/fetch_recorded_demo_1730997119.txt'
    data = np.loadtxt(path)  # load the file into an array

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    # x = np.linspace(0, 10).reshape((50, 1))
    # y = np.sin(x) + 3 * np.cos(x) ** 2
    # z = 0.005 * (x - 5) ** 4 - 0.05 * (x - 5) ** 2

    traj_list = [x, y, z]
    traj = np.hstack((x, y, z))
    plot_with_slider_1D(traj_list)
    plot_with_slider_3D(traj)

