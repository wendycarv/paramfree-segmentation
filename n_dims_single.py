import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import mode
import itertools
from scipy import signal
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties


np.set_printoptions(legacy='1.25')

def read_data(fname):
    with h5py.File(fname, 'r') as hf:
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
        
        gp = hf.get('gripper_info')
        gripper_time = np.array(gp.get('gripper_time'))
        gripper_pos = np.array(gp.get('gripper_position'))
        gripper_data = [gripper_time, gripper_pos]
    
    return joint_data, tf_data, gripper_data

def SSE(x, y):
    return np.linalg.norm(x - y)**2

# full task data, one subtask example
def autocorr(data, example):
    n_pts, n_dims = np.shape(data)
    ex_pts, _ = np.shape(example)
    dists = np.zeros((n_pts - ex_pts,))
    for i in range(n_pts - ex_pts):
        dist = SSE(data[i:i+ex_pts, :], example)
        dists[i] = dist
    return dists

# data: full task x; examples: 5
# perform sse in n-dim
def autocorr_examples(data, examples):
    dists = []
    for ex in examples:
        dists.append(autocorr(data, ex))
    return dists

def process_demo_xyz(xyz_file, h5_file):
    xyz_data = np.loadtxt(xyz_file)
    return xyz_data

def process_demo_with_smoothing(xyz_file, window_length, polyorder):
    xyz_data = np.loadtxt(xyz_file)

    # applying savgol filter to x, y, and z
    smoothed_x = signal.savgol_filter(xyz_data[:, 0], window_length, polyorder)
    smoothed_y = signal.savgol_filter(xyz_data[:, 1], window_length, polyorder)
    smoothed_z = signal.savgol_filter(xyz_data[:, 2], window_length, polyorder)

    # put dimensions back into a single array
    smoothed_xyz_data = np.vstack((smoothed_x, smoothed_y, smoothed_z)).T
    return smoothed_xyz_data

def is_sorted_ascending(lst):
    return all(earlier <= later for earlier, later in zip(lst, lst[1:]))

def accuracy_score(truth, predictions):
    differences = [abs(c - p) for c, p in zip(truth, predictions)]
    percentage_differences = [(diff / c) * 100 if c != 0 else 0 for diff, c in zip(differences, truth)]
    mean_percentage_difference = sum(percentage_differences) / len(percentage_differences)
    return mean_percentage_difference

def optimize_smoothing(full_task_xyz, best_combination, correct_indices, window_lengths, polyorders, xyz_dir):
    smooth_fulltask = np.copy(full_task_xyz)
    best_params = None
    lowest_mean_diff = float('inf')
    best_predicted_indices = []

    print("performing optimized smoothing...")
    for window_length in window_lengths:
        for polyorder in polyorders:
            if polyorder < (window_length - 2):
                classes_xyz = []
                for file in best_combination:
                    xyz_path = xyz_dir + "subtasks/" + file + ".txt"
                    classes_xyz.append(process_demo_with_smoothing(xyz_path, window_length, polyorder))

            for i in range(3):  # iterate over x, y, z dimensions
                smooth_fulltask[:, i] = signal.savgol_filter(full_task_xyz[:, i], window_length, polyorder)

            xyz_dists_per_class = autocorr_examples(full_task_xyz, classes_xyz)
            predicted_indices = [np.argmin(dist) for dist in xyz_dists_per_class]

            mean_diff_percentage = accuracy_score(correct_indices, predicted_indices)
            #print(f"Mean difference percentage found: {mean_diff_percentage}, at window length {window_length}, poly order {polyorder}")
            # prioritize simply best mean diff 
            if mean_diff_percentage < lowest_mean_diff:
                lowest_mean_diff = mean_diff_percentage
                best_predicted_indices = predicted_indices
                best_params = (window_length, polyorder)      

    return best_params, best_predicted_indices, lowest_mean_diff


def main():
    # base directories
    #velocity_dir = "/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/fetch_table_demos/velocities/"
    xyz_dir = "/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/fetch_table_demos/xyz data/"
    h5_dir = "/Users/wendy/Desktop/school/uml/robotics/auto_correlation/raw data/fetch_table_demos/h5 files/"
    
    # put best found files here
    files = [
        "fetch_recorded_demo_1730996323",
        "fetch_recorded_demo_1730996527",
        "fetch_recorded_demo_1730996653",
        "fetch_recorded_demo_1730996760",
        "fetch_recorded_demo_1730996961"
    ]

    classes_xyz = []

    i=0
    for file in files:
        xyz_path = xyz_dir + "subtasks/" + file + ".txt"
        h5_path = h5_dir + file + ".h5"
        i += 1
        classes_xyz.append(process_demo_xyz(xyz_path, h5_path))

    # replace with full task file name
    full_task_xyz = process_demo_xyz(xyz_dir + "full_tasks/fetch_recorded_demo_1730997119.txt", h5_dir + "fetch_recorded_demo_1730997119.h5")

    xyz_dists_per_class = autocorr_examples(full_task_xyz, classes_xyz)
    best_predicted_indices = [np.argmin(dist) for dist in xyz_dists_per_class]

    # need to be replaced with each task (see demos spreadsheet for these)
    correct_indices = [0, 1125, 2591, 3986, 5666]

    mean_diff_percentage = accuracy_score(correct_indices, best_predicted_indices)

    # output
    print(f"Lowest mean difference: {mean_diff_percentage:.2f}%")
    print(f"Predicted indices: {best_predicted_indices}")

    '''
    # apply smoothing factor
    window_lengths = range(51, 350, 10)
    polyorders = range(2, 5)

    best_params, best_predicted_indices, new_mean_diff = optimize_smoothing(full_task_xyz, files, correct_indices, window_lengths, polyorders, xyz_dir)

    # output
    print("WITH SMOOTHING:\n")
    print(f"Lowest Mean Difference: {new_mean_diff:.2f}%")
    print(f"Predicted Indices: {best_predicted_indices}")
    print(f"Best smoothing parameters: {best_params}")
    '''
    n_classes = len(classes_xyz)
    for c in range(n_classes):
        if c == 0:
            if best_predicted_indices[c] != 0:
                best_predicted_indices[c] = 0
        elif c != n_classes - 1:
            if best_predicted_indices[c] != best_predicted_indices[c+1] - 1:
                avg = (best_predicted_indices[c+1] - best_predicted_indices[c]) / 2
                #best_predicted_indices[c+1] -= avg
                #best_predicted_indices[c] += avg
    
    lengths = []
    for i in range(n_classes):
        if i != n_classes - 1:
            lengths.append(best_predicted_indices[i+1]-best_predicted_indices[i])
        else:
            lengths.append(len(full_task_xyz)-best_predicted_indices[i])

    print("lengths: ", lengths)
    print("FULL LENGTH", len(full_task_xyz))

       # plotting the data for best combo
    colors = ['r', 'g', 'b', 'm', 'c']
    dims = ["x", "y", "z"]
    linestyles = ["-", ":", "--"]
    fig = plt.figure(figsize=(9, 7))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.2, hspace=0.15)
    ax1 = plt.subplot(212)
    ax1.set_title('Segmented', family='Times New Roman',fontsize=16)
    for i in range(n_classes):  # plotting each class
        for j in range(classes_xyz[i].shape[1]):
        #ax1.plot(np.arange(best_predicted_indices[i], best_predicted_indices[i]+len(best_classes_xyz[i])), 
                 #full_task_xyz[best_predicted_indices[i]:best_predicted_indices[i]+len(best_classes_xyz[i])], 
                 #colors[i])
            #ax1.plot(np.arange(best_predicted_indices[i], best_predicted_indices[i]+len(best_classes_xyz[i])), 
            #        full_task_xyz[best_predicted_indices[i]:best_predicted_indices[i]+len(best_classes_xyz[i])][:,j], 
            #        color=colors[i], linewidth=2, linestyle=linestyles[j])
            if i < n_classes - 1:
                ax1.plot(np.arange(best_predicted_indices[i], best_predicted_indices[i]+(best_predicted_indices[i+1]-best_predicted_indices[i])), 
                    full_task_xyz[best_predicted_indices[i]:best_predicted_indices[i]+(best_predicted_indices[i+1]-best_predicted_indices[i])][:,j], 
                    color=colors[i], linewidth=2, linestyle=linestyles[j])
            else:
                ax1.plot(np.arange(best_predicted_indices[i], best_predicted_indices[i]+lengths[i]), 
                    full_task_xyz[best_predicted_indices[i]:best_predicted_indices[i]+lengths[i]][:,j], 
                    color=colors[i], linewidth=2, linestyle=linestyles[j])
                
    #ax1.tick_params(axis='both', which='both', left=False, bottom=False, top=False, length=0, labelleft=False, labelbottom=False) # labels along the bottom edge are off
    ax1.tick_params(axis='both', which='both', left=True, bottom=True, top=False, length=2, labelleft=False, labelbottom=False) # labels along the bottom edge are off
    for i in range(n_classes):
        ax2 = plt.subplot(2, n_classes, i+1)
        ax2.set_title(f'Class {i+1}', family='Times New Roman', fontsize=16)
        for dim in range(classes_xyz[i].shape[1]):
            ax2.plot(classes_xyz[i][:, dim], color=colors[i], linewidth=2, linestyle=linestyles[dim])
            ax2.tick_params(axis='both', which='both', left=True, bottom=True, top=False, length=2, labelleft=False, labelbottom=False) # labels along the bottom edge are off
    legend_elements = [Line2D([0], [0], color='black', lw=2, label='x', linestyle='-'),
                   Line2D([0], [0], color='black', lw=2, label='y', linestyle=':'),
                   Line2D([0], [0], color='black', lw=2, label='z', linestyle='--')]
    ax2.legend(handles=legend_elements, ncol=3, loc='lower center', bbox_to_anchor=(-1.9, -1.4), prop={'family':'Times New Roman', 'size':16})

    plt.savefig("test.svg", dpi=300)

    plt.figure(figsize=(15, 7))
    ax1 = plt.subplot(212)
    ax1.set_title('Original',family='Times New Roman')
    # plotting original x y and z
    for i in range(len(dims)):
        ax1.plot(full_task_xyz[:,i], color='black',linestyle=linestyles[i])
    ax1.tick_params(axis='both', which='both', left=False, bottom=False, top=False, length=0, labelleft=False, labelbottom=False) # labels along the bottom edge are off
    for i in range(n_classes):
        ax2 = plt.subplot(2, n_classes, i+1)
        ax2.set_title(f'Class {i+1}', family='Times New Roman')
        for dim in range(classes_xyz[i].shape[1]):
            ax2.plot(classes_xyz[i][:, dim], color=colors[i], linestyle=linestyles[dim])
    ax2.tick_params(axis='both', which='both', left=False, bottom=False, top=False, length=0, labelleft=False, labelbottom=False) # labels along the bottom edge are off
    legend_elements2 = [Line2D([0], [0], color='black', lw=1, label='x', linestyle='-'),
                   Line2D([0], [0], color='black', lw=1, label='y', linestyle=':'),
                   Line2D([0], [0], color='black', lw=1, label='z', linestyle='--')]
    ax2.legend(handles=legend_elements2, loc='center left', bbox_to_anchor=(1.1, -0.1), prop={'family':'Times New Roman'})
    
    
    plt.show()

if __name__ == "__main__":
    main()
