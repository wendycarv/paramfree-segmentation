import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import mode

np.set_printoptions(legacy='1.25')

def read_data(fname):
    hf = h5py.File(fname, 'r')
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
    
    hf.close()
    
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

def process_demo(velocity_file, h5_file, joint_num):
    # load velocity data from the text file
    velocity_data = np.loadtxt(velocity_file)
    
    joint_data, tf_data, gripper_data = read_data(h5_file)
    time_data = joint_data[0][:, 0] + joint_data[0][:, 1] * 1e-9
    
    joint = velocity_data[:, joint_num]
    joint_acc = np.gradient(joint, time_data)
    joint_acc = joint_acc[:, np.newaxis]
    joint = joint[:, np.newaxis]
    return joint_acc

def process_demo_xyz(xyz_file, h5_file):
    xyz_data = np.loadtxt(xyz_file)
    
    #joint_data, tf_data, gripper_data = read_data(h5_file)
    
    #pos = xyz_data[:, pos_num]
    #pos = pos[:, np.newaxis]
    return xyz_data

def process_demo_full(velocity_file, h5_file, joint_num):
    # load velocity data from the text file
    velocity_data = np.loadtxt(velocity_file)
    
    joint_data, tf_data, gripper_data = read_data(h5_file)
    time_data = joint_data[0][:, 0] + joint_data[0][:, 1] * 1e-9
    
    joint = velocity_data[:, joint_num]
    joint_acc = np.gradient(joint, time_data)
    joint_acc = joint_acc[:, np.newaxis]
    joint = joint[:, np.newaxis]
    return joint_acc, joint

def accuracy_score(truth, predictions):

    y = 0
    _y = 0

    # Calculate the absolute differences between corresponding indices
    differences = [abs(c - p) for c, p in zip(truth, predictions)]

    # Calculate the percentage differences relative to the correct indices
    percentage_differences = [(diff / c) * 100 if c != 0 else 0 for diff, c in zip(differences, truth)]

    # Calculate the mean percentage difference
    mean_percentage_difference = sum(percentage_differences) / len(percentage_differences)

    return mean_percentage_difference


def main():
    files = [
        "fetch_recorded_demo_1730996356",
        "fetch_recorded_demo_1730996415",
        "fetch_recorded_demo_1730996653",
        "fetch_recorded_demo_1730996811",
        "fetch_recorded_demo_1730996961",
    ]
    # base directories
    velocity_dir = "/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/velocities/"
    xyz_dir = "/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/xyz data/"
    h5_dir = "/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/h5 files/"

    '''
    num_joints = 6
    select_joints = [[] for _ in range(num_joints)]
    # process each file
    for file in files:
        velocity_path = velocity_dir + file + ".txt"
        h5_path = h5_dir + file + ".h5"
        for joint_num in range(num_joints):  # grabbing joint acceleration
            select_joints[joint_num].append(process_demo(velocity_path, h5_path, joint_num))will 
    '''
    
    # *** XYZ DATA ***
    n_dim = 3
    classes_xyz = []
    # process each file
    i = 1
    for file in files:
        xyz_path = xyz_dir + "subtasks/" + file + ".txt"
        h5_path = h5_dir + file + ".h5"
        i += 1
        classes_xyz.append(process_demo_xyz(xyz_path, h5_path))

    print(classes_xyz)
    full_task_xyz = process_demo_xyz(xyz_dir + "full_tasks/fetch_recorded_demo_1730997119.txt", h5_dir + "fetch_recorded_demo_1730997119.h5")

    predicted_indices = []
    xyz_dists_per_class = autocorr_examples(full_task_xyz, classes_xyz)
    predicted_indices = [np.argmin(dist) for dist in xyz_dists_per_class]

    print("Predictions:", predicted_indices)

    # get accuracy
    # #1: [0, 1125, 2591, 3986, 5666] vs [46, 622, 2877, 4476, 6050]
    correct_indices = [0, 1125, 2591, 3986, 5666]  # for full task #1

    mean_diff_percentage = accuracy_score(correct_indices, predicted_indices)
    print(f"Mean difference: {mean_diff_percentage:.2f}%")

 
    # *********** PLOTTING THE DATA *************** 

    n_classes = len(classes_xyz)
    colors = ['r', 'g', 'b', 'm', 'c']
    dims = ["x", "y", "z"]
    plt.figure(figsize=(15, 7))
    ax1 = plt.subplot(212)
    ax1.set_title('Segmented')
    for i in range(n_classes):
        ax1.plot(np.arange(predicted_indices[i], predicted_indices[i]+len(classes_xyz[i])), full_task_xyz[predicted_indices[i]:predicted_indices[i]+len(classes_xyz[i])], colors[i])

    for i in range(n_classes):
        ax2 = plt.subplot(2, n_classes, i+1)
        ax2.set_title(f'Class {i+1}')
        ax2.plot(classes_xyz[i], 'k')

    plt.figure(figsize=(15, 7))
    ax1 = plt.subplot(212)
    ax1.set_title('Original')
    for i in range(n_dim):
        ax1.plot(full_task_xyz[:,i], colors[i], label=f"{dims[i]}")

    plt.legend()

    for i in range(n_classes):
        ax2 = plt.subplot(2, n_classes, i+1)
        ax2.set_title(f'Class {i+1}')
        ax2.plot(classes_xyz[i], 'k')

    plt.show()

if __name__ == "__main__":
    main()