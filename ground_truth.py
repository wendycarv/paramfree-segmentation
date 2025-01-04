import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import h5py
import glob
import os
def plot_with_slider_1D(data, time, tf_data):
    cur_idx = 0  # Start at the first index
    time = time % 10000
    # normalize to start from 0
    time = time - time[0]
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)

    idxs = []  # To hold the marker plots for each sequence
    labels = ['x', 'y', 'z']
    colors = ["red", "blue", "green"]

    # Plot each array in `data`
    for d in range(len(data)):
        ax.plot(range(len(data[d])), data[d], label=f'{labels[d]}', color=colors[d], alpha=0.5)
        idx_marker, = ax.plot([cur_idx], [data[d][cur_idx]], '.', ms=9, color=colors[d], label=f'{labels[d]}')
        idxs.append(idx_marker)

    # Slider to select index
    axidx = plt.axes([0.25, 0.15, 0.65, 0.03])
    idx_slider = Slider(axidx, 'index', 0, len(data[0]) - 1, valinit=cur_idx, valfmt='%0.0f')

    # Secondary x-axis for time
    ax2 = ax.twiny()

    # Plot the time-transformed data
    for d in range(3):
        ax2.plot(time, tf_data[1][:, d], color=colors[d], alpha=0)

    ax2.set_xlabel('time')
    time_ticks = np.linspace(time[0], time[-1], 10)
    ax2.set_xticks(time_ticks)
    # Time slider
    axtime = plt.axes([0.25, 0.05, 0.65, 0.03])
    time_slider = Slider(axtime, 'time', time[0], time[-1], valinit=cur_idx, valfmt='%0.2f')

    def update(val):
        """Update markers and synchronize the sliders."""
        nonlocal cur_idx
        cur_idx = int(idx_slider.val)
        for d in range(len(data)):
            idxs[d].set_xdata([cur_idx])  # Update x position
            idxs[d].set_ydata([data[d][cur_idx]])  # Update y position
        # Update time slider position
        time_slider.set_val(time[cur_idx])

    def update_time(val):
        """Update markers and synchronize the index slider."""
        nonlocal cur_idx
        cur_time = val
        cur_idx = np.argmin(np.abs(time - cur_time))  # Find closest index to the time value
        for d in range(len(data)):
            idxs[d].set_xdata([cur_idx])  # Update x position
            idxs[d].set_ydata([data[d][cur_idx]])  # Update y position
        # Update index slider position
        idx_slider.set_val(cur_idx)

    idx_slider.on_changed(update)
    time_slider.on_changed(update_time)

    ax.legend()
    ax.set_xlabel('index')
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.show()


def plot_with_slider_3D(data):
    cur_idx = 0
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.subplots_adjust(bottom=0.35)
    ax.plot3D(data[:, 0], data[:, 1], data[:, 2], 'k', lw=3)
    dot, = ax.plot3D(data[cur_idx, 0], data[cur_idx, 1], data[cur_idx, 2], 'k.', ms=9, color='red')

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


def read_data(fname):
    hf = h5py.File(fname, 'r')
    #print(list(hf.keys()))
    js = hf.get('joint_state_info')
    joint_time = np.array(js.get('joint_time'))
    joint_pos = np.array(js.get('joint_positions'))
    joint_vel = np.array(js.get('joint_velocities'))
    joint_eff = np.array(js.get('joint_effort'))
    joint_data = [joint_time, joint_pos, joint_vel, joint_eff]

    tf = hf.get('transform_info')
    tf_time = np.array(tf.get('transform_time'))
    tf_pos = np.array(tf.get('transform_positions'))
    tf_rot = np.array(tf.get('transform_orientations'))
    tf_data = [tf_time, tf_pos, tf_rot]
    #print(tf_pos)

    # wr = hf.get('wrench_info')
    # wrench_time = np.array(wr.get('wrench_time'))
    # wrench_frc = np.array(wr.get('wrench_force'))
    # wrench_trq = np.array(wr.get('wrench_torque'))
    # wrench_data = [wrench_time, wrench_frc, wrench_trq]

    gp = hf.get('gripper_info')
    gripper_time = np.array(gp.get('gripper_time'))
    gripper_pos = np.array(gp.get('gripper_position'))
    gripper_data = [gripper_time, gripper_pos]

    hf.close()

    # return joint_data, tf_data, wrench_data, gripper_data
    return joint_data, tf_data, gripper_data

if __name__ == '__main__':
    path = '/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/xyz data/full_tasks/fetch_recorded_demo_1730997956.txt'
    h5_path = '/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/h5 files/fetch_recorded_demo_1730997956.h5'
    data = np.loadtxt(path)  # load the file into an array
    '''
    x = np.linspace(0, 10).reshape((50, 1))
    y = np.sin(x) + 3 * np.cos(x) ** 2
    z = 0.005 * (x - 5) ** 4 - 0.05 * (x - 5) ** 2
    '''
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    joint_data, tf_data, gripper_data = read_data(h5_path)
    time = tf_data[0][:, 0] + tf_data[0][:, 1] * (10.0 ** -9)

    traj_list = [x, y, z]
    #traj = np.hstack((x, y, z))
    traj = data
    plot_with_slider_1D(traj_list, time, tf_data)
    plot_with_slider_3D(traj)

    # plot the main task
    plt.plot(time, tf_data[1][:, 0], label="x", color="red", alpha=0.5)
    plt.plot(time, tf_data[1][:, 1], label="y", color="blue", alpha=0.5)
    plt.plot(time, tf_data[1][:, 2], label="z", color="green", alpha=0.5)

    plt.xlabel("time")
    plt.ylabel("position")

    '''
    # Highlight the matching segment
    matching_time_steps = np.arange(start_idx, start_idx + subtask.shape[1])
    plt.plot(matching_time_steps, matching_segment[0, :], label="Matching Segment - X", color="red")
    plt.plot(matching_time_steps, matching_segment[1, :], label="Matching Segment - Y", color="blue")
    plt.plot(matching_time_steps, matching_segment[2, :], label="Matching Segment - Z", color="green")

    # Add vertical lines for the subtask boundaries
    plt.axvline(start_idx, color="black", linestyle="--", label="Subtask Start")
    #plt.axvline(end_idx, color="gray", linestyle="--", label="Subtask End")
    '''

    plt.title("Ground truth for fetch task")
    plt.legend()
    # plt.grid()
    plt.show()
