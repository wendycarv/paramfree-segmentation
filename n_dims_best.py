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
    #print("RANGE: ", range(n_pts-ex_pts))
    for i in range(n_pts - ex_pts):
        #print(data[i:i+ex_pts, :])
        dist = SSE(data[i:i+ex_pts, :], example)
        dists[i] = dist
        #print(dist)
    #print(dists.shape)
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

def is_sorted_ascending(lst):
    return all(earlier <= later for earlier, later in zip(lst, lst[1:]))

def accuracy_score(truth, predictions):
    differences = [abs(c - p) for c, p in zip(truth, predictions)]
    percentage_differences = [(diff / c) * 100 if c != 0 else 0 for diff, c in zip(differences, truth)]
    mean_percentage_difference = sum(percentage_differences) / len(percentage_differences)
    return mean_percentage_difference

def main():
    # base directories
    velocity_dir = "/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/velocities/"
    xyz_dir = "/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/xyz data/"
    h5_dir = "/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/h5 files/"
    
    # subclass file names
    subclasses = [
        ["fetch_recorded_demo_1730996265", "fetch_recorded_demo_1730996323", "fetch_recorded_demo_1730996356", "fetch_recorded_demo_1730996383"],
        ["fetch_recorded_demo_1730996415", "fetch_recorded_demo_1730996496", "fetch_recorded_demo_1730996527", "fetch_recorded_demo_1730996563"],
        ["fetch_recorded_demo_1730996603", "fetch_recorded_demo_1730996653", "fetch_recorded_demo_1730996691", "fetch_recorded_demo_1730996720"],
        ["fetch_recorded_demo_1730996760", "fetch_recorded_demo_1730996811", "fetch_recorded_demo_1730996844", "fetch_recorded_demo_1730996879"],
        ["fetch_recorded_demo_1730996917", "fetch_recorded_demo_1730996961", "fetch_recorded_demo_1730996997", "fetch_recorded_demo_1730997035"],
    ]

    # create combinations of one file from each subclass
    all_combinations = list(itertools.product(*subclasses))

    # placeholders
    best_combination = None
    best_mean_diff = float('inf')
    best_predicted_indices = []

    # process each combination and find best one
    for combination in all_combinations:
        classes_xyz = []
        for file in combination:
            xyz_path = xyz_dir + "subtasks/" + file + ".txt"
            h5_path = h5_dir + file + ".h5"
            classes_xyz.append(process_demo_xyz(xyz_path, h5_path))

        #print("size:", classes_xyz[0].shape[1])
        full_task_xyz = process_demo_xyz(xyz_dir + "full_tasks/fetch_recorded_demo_1730997119.txt", h5_dir + "fetch_recorded_demo_1730997119.h5")

        xyz_dists_per_class = autocorr_examples(full_task_xyz, classes_xyz)
        #print(len(xyz_dists_per_class[0]))
        predicted_indices = [np.argmin(dist) for dist in xyz_dists_per_class]

        correct_indices = [0, 1125, 2591, 3986, 5666] # Replace with your actual correct indices

        mean_diff_percentage = accuracy_score(correct_indices, predicted_indices)
        #print(combination, best_mean_diff)
        #print(mean_diff_percentage)
        # prioritize simply best mean diff 

        # prefer predictions that show indices in correct order
        if is_sorted_ascending(predicted_indices):
            mean_diff_percentage = accuracy_score(correct_indices, predicted_indices)

            if mean_diff_percentage < best_mean_diff:
                best_mean_diff = mean_diff_percentage
                best_combination = combination
                best_predicted_indices = predicted_indices
                best_classes_xyz = classes_xyz
        '''

        if mean_diff_percentage < best_mean_diff:
            best_mean_diff = mean_diff_percentage
            best_combination = combination
            best_predicted_indices = predicted_indices
            best_classes_xyz = classes_xyz      


        '''

    print(f"Best Combination: {best_combination}")
    print(f"Lowest Mean Difference: {best_mean_diff:.2f}%")
    print(f"Predicted Indices: {best_predicted_indices}")

    # now that we have the predicted indices, need to try to fill in gaps
    # for i in predicted_indices
    # if p[0] != 0, make it = 0
    # if p[1] != p[2] - 1, then find the average between the two: p[2] - p[1] = x. p[1] += x, p[2] -= x;
    # if p[4] != len(full_task_xyz) then make it so
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
    # output


    '''
    # what about now that we've changed the predicted indices?
    mean_diff_percentage = accuracy_score(correct_indices, best_predicted_indices)

    print(f"Best Combination: {best_combination}")
    print(f"New Mean Difference: {mean_diff_percentage:.2f}%")
    print(f"Predicted Indices: {best_predicted_indices}")
    '''
    # plotting the data for best combo
    n_classes = len(best_combination)
    colors = ['r', 'g', 'b', 'm', 'c']
    dims = ["x", "y", "z"]
    linestyles = ["-", ":", "--"]
    plt.figure(figsize=(9, 7))
    ax1 = plt.subplot(212)
    ax1.set_title('Segmented', family='Times New Roman')
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
    for i in range(n_classes):
        ax2 = plt.subplot(2, n_classes, i+1)
        ax2.set_title(f'Class {i+1}', family='Times New Roman')
        for dim in range(classes_xyz[i].shape[1]):
            ax2.plot(classes_xyz[i][:, dim], color=colors[i], linewidth=2, linestyle=linestyles[dim])
            #ax2.tick_params(axis='both', which='both', left=False, bottom=False, top=False, length=0, labelleft=False, labelbottom=False) # labels along the bottom edge are off
    legend_elements = [Line2D([0], [0], color='black', lw=2, label='x', linestyle='-'),
                   Line2D([0], [0], color='black', lw=2, label='y', linestyle=':'),
                   Line2D([0], [0], color='black', lw=2, label='z', linestyle='--')]
    ax2.legend(handles=legend_elements, ncol=3, loc='lower center', bbox_to_anchor=(-1.7, -1.5), prop={'family':'Times New Roman'})

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
    #plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
