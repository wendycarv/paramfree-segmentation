import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.signal import savgol_filter

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


def min_max_normalize(data, all_data):
    data_min = np.min(all_data, axis=0)
    data_max = np.max(all_data, axis=0)
    return (data - data_min) / (data_max - data_min)

def z_score_normalize(data, all_data):
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (data - mean) / std

def normalize_per_feature(data, all_data):
    for i in range(data.shape[1]):  
        mean = np.mean(all_data[:, i])
        std = np.std(all_data[:, i])
        if std == 0:
            std = 1  # Avoid division by zero
        data[:, i] = (data[:, i] - mean) / std
    return data

def robust_normalize(data, all_data):
    median = np.median(all_data, axis=0)
    q1 = np.percentile(all_data, 25, axis=0)
    q3 = np.percentile(all_data, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1  # Avoid division by zero
    return (data - median) / iqr

def evaluate_normalization(full_task_xyz, classes_xyz, normalize_fn):

    all_data = np.vstack([full_task_xyz] + classes_xyz)
    
    # Apply selected normalization function
    full_task_xyz_norm = normalize_fn(full_task_xyz, all_data)
    classes_xyz_norm = [normalize_fn(ex, all_data) for ex in classes_xyz]

    # Compute autocorrelation distances
    xyz_dists_per_class = autocorr_examples(full_task_xyz_norm, classes_xyz_norm)
    predicted_indices = [np.argmin(dist) for dist in xyz_dists_per_class]

    # Compare predictions to correct indices
    correct_indices = [0, 1125, 2591, 3986, 5666]
    mean_diff_percentage = accuracy_score(correct_indices, predicted_indices)

    return mean_diff_percentage

def find_best_normalization(full_task_xyz, classes_xyz):
    normalizations = {
        "Min-Max": min_max_normalize,
        "Z-Score": z_score_normalize,
        "Feature-wise Z-Score": normalize_per_feature,
        "Robust Scaling": robust_normalize
    }

    best_method = None
    best_score = float("inf")  # We want the **lowest mean difference percentage**
    
    for name, func in normalizations.items():
        score = evaluate_normalization(full_task_xyz, classes_xyz, func)
        print(f"{name} normalization → Mean Difference: {score:.2f}%")
        
        if score < best_score:
            best_score = score
            best_method = name

    print(f"\nBest Normalization: {best_method} with {best_score:.2f}% Mean Difference")
    return normalizations[best_method]  # Return best function



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

def accuracy_score(truth, predictions) :
    # Calculate the absolute differences between corresponding indices
    differences = [abs (c - p) for c, p in zip(truth, predictions)]
    # Calculate the percentage differences relative to the correct indices
    percentage_differences = [(diff / c) * 100 if c != 0 else 0 for diff, c in zip(differences, truth)]
    # Calculate the mean percentage difference
    mean_percentage_difference = sum (percentage_differences) / len(percentage_differences)
    return mean_percentage_difference


def main():
    files = [
        "fetch_recorded_demo_1730996323",
        "fetch_recorded_demo_1730996415",
        "fetch_recorded_demo_1730996653",
        "fetch_recorded_demo_1730996844",
        "fetch_recorded_demo_1730996917",
    ]
    # base directories
    velocity_dir = "/Users/meriemelkoudi/Desktop/Pearl Lab/auto-correlation/table demos/velocities/"
    xyz_dir = "/Users/meriemelkoudi/Desktop/Pearl Lab/auto-correlation/table demos/xyz data/"
    h5_dir = "/Users/meriemelkoudi/Desktop/Pearl Lab/auto-correlation/table demos/h5 files/"

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
   # full_task_xyz = process_demo_xyz(xyz_dir + "full_tasks/fetch_recorded_demo_1730997119.txt", h5_dir + "fetch_recorded_demo_1730997119.h5")
    full_task_xyz = process_demo_xyz(xyz_dir + "full_tasks/fetch_recorded_demo_1730997956.txt", h5_dir + "fetch_recorded_demo_1730997956.h5")
    predicted_indices = []

     # pick the best normalization method
    best_normalization = find_best_normalization(full_task_xyz, classes_xyz)

    # normalization
    all_data = np.vstack([full_task_xyz] + classes_xyz)
    full_task_xyz_normalized = best_normalization(full_task_xyz, all_data)
    classes_xyz_normalized = [best_normalization(ex, all_data) for ex in classes_xyz]


    # do the autocorr on normalized data :)
    xyz_dists_per_class = autocorr_examples(full_task_xyz_normalized, classes_xyz_normalized)
    predicted_indices = [np.argmin(dist) for dist in xyz_dists_per_class]

    print("Predictions:", predicted_indices)

    # get accuracy
    # #1: [0, 1125, 2591, 3986, 5666] vs [46, 622, 2877, 4476, 6050]
    correct_indices = [0, 1898, 4081, 5442, 6829]  # for full task #1

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


    '''
    #xyz_dists_per_class = autocorr_examples(full_task_xyz, classes_xyz)
    #predicted_indices = [np.argmin(dist) for dist in xyz_dists_per_class]

    # normalize!!
    all_data = np.vstack([full_task_xyz] + classes_xyz)  # Combine all data
    full_task_xyz_normalized = robust_normalize(full_task_xyz, all_data)
    classes_xyz_normalized = [robust_normalize(ex, all_data) for ex in classes_xyz]
    '''