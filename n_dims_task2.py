import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy import signal

#np.set_printoptions(legacy='1.25')
plt.rcParams["font.family"] = "Times New Roman"


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

def process_demo_xyz(xyz_file, h5_file):
    data = np.loadtxt(xyz_file, dtype=float)  # Explicitly set dtype
    print("Loaded data:", data[:5])  # Print first few values
    print("Shape: ", data.shape)
    return data

def accuracy_score(truth, predictions):

    # Calculate the absolute differences between corresponding indices
    differences = [abs(c - p) for c, p in zip(truth, predictions)]

    # Calculate the percentage differences relative to the correct indices
    percentage_differences = [(diff / c) * 100 if c != 0 else 0 for diff, c in zip(differences, truth)]

    # Calculate the mean percentage difference
    mean_percentage_difference = sum(percentage_differences) / len(percentage_differences)

    return mean_percentage_difference


def return_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        if "smoothed" not in hf:
            print("No 'smoothed' group found in the file.")
            return

        # need to find min/max values of y for flipping
        all_y_values = []
        for key in hf["smoothed"].keys():
            y_data = np.array(hf[f"smoothed/{key}/y"])
            all_y_values.extend(y_data)
        
        y_min, y_max = min(all_y_values), max(all_y_values)

        # iterate over each stroke
        for key in sorted(hf["smoothed"].keys(), key=int):
            x_data = np.array(hf[f"smoothed/{key}/x"])
            y_data = np.array(hf[f"smoothed/{key}/y"])
            t_data = np.array(hf[f"smoothed/{key}/t"])

            # sort by time
            sorted_indices = np.argsort(t_data)
            x_data = x_data[sorted_indices]
            y_data = y_data[sorted_indices]

            # reverse according to max y value
            y_data = y_max - (y_data - y_min)

            return x_data, y_data


def main():
    files = [
        "class1",
        "class2",
        "class3",
    ]
    # base directories
    xyz_dir = "/Users/wendy/Desktop/school/uml/robotics/dog/xy data/"
    h5_dir = "/Users/wendy/Desktop/school/uml/robotics/dog/h5 files/"

    # *** XYZ DATA ***
    n_dim = 2
    classes_xyz = []
    # process each file
    i = 1
    print("processing sub-task classes...")
    for file in files:
        xyz_path = xyz_dir + file + ".txt"
        h5_path = h5_dir + file + ".h5"
        i += 1
        classes_xyz.append(process_demo_xyz(xyz_path, h5_path))

    print("processing full-task ...")
    full_task_xyz = process_demo_xyz(xyz_dir + "dog3.txt", h5_dir + "dog3.h5")
    print(full_task_xyz.shape)
    predicted_indices = []
    xyz_dists_per_class = autocorr_examples(full_task_xyz, classes_xyz)
    #print(xyz_dists_per_class)
    predicted_indices = [np.argmin(dist) for dist in xyz_dists_per_class]

    print("Predictions:", predicted_indices)
    # get accuracy
    # #1: [0, 1125, 2591, 3986, 5666] vs [46, 622, 2877, 4476, 6050]
    #correct_indices = [0, 1125, 2591, 3986, 5666] # for task 3
    
    # [0, 1812, 3844, 5732, 7090] # for task 2
    
    # [0, 1125, 2591, 3986, 5666]  # for full task #1

    #mean_diff_percentage = accuracy_score(correct_indices, predicted_indices)
    #print(f"Mean difference: {mean_diff_percentage:.2f}%")

 
    # *********** PLOTTING THE DATA *************** 

    full_x_data, full_y_data = return_data("/Users/wendy/Desktop/school/uml/robotics/dog/h5 files/dog3.h5")
    n_classes = len(classes_xyz)
    colors = ['r', 'g', 'b']
    dims = ["x", "y"]
    full_x_data = []
    full_y_data = []

    plt.figure(figsize=(8, 7))
    ax1 = plt.subplot(212)
    ax1.set_title('Original')



    # ***** NEEDS TO BE REORGANIZED, FOR NOW THIS WORKS ***** 
    # essentially doing same thing as what was done in return_data()
    with h5py.File("/Users/wendy/Desktop/school/uml/robotics/dog/h5 files/dog3.h5", 'r') as hf:

        all_y_values = []
        for key in hf["smoothed"].keys():
            y_data = np.array(hf[f"smoothed/{key}/y"])
            all_y_values.extend(y_data)
        
        y_min, y_max = min(all_y_values), max(all_y_values)

        for key in sorted(hf["smoothed"].keys(), key=int): 
            x_data = np.array(hf[f"smoothed/{key}/x"])
            y_data = np.array(hf[f"smoothed/{key}/y"])
            t_data = np.array(hf[f"smoothed/{key}/t"])

            sorted_indices = np.argsort(t_data)
            x_data = x_data[sorted_indices]
            y_data = y_data[sorted_indices]

            y_data = y_max - (y_data - y_min)

            full_x_data.extend(x_data) 
            full_y_data.extend(y_data)

            # adding nan to stop from adding random connections between each letter
            full_x_data.append(np.nan)
            full_y_data.append(np.nan)

            ax1.plot(x_data, y_data, linewidth=3, color="black")

    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")

    i = 0

    for file in files:
        ax2 = plt.subplot(2, n_classes, i+1)
        ax2.set_title(f'Class {i+1}')
        with h5py.File(f"/Users/wendy/Desktop/school/uml/robotics/dog/h5 files/{file}.h5", 'r') as hf:

            all_y_values = []
            for key in hf["smoothed"].keys():
                y_data = np.array(hf[f"smoothed/{key}/y"])
                all_y_values.extend(y_data)
            
            y_min, y_max = min(all_y_values), max(all_y_values)

            for key in sorted(hf["smoothed"].keys(), key=int):
                x_data = np.array(hf[f"smoothed/{key}/x"])
                y_data = np.array(hf[f"smoothed/{key}/y"])
                t_data = np.array(hf[f"smoothed/{key}/t"])

                sorted_indices = np.argsort(t_data)
                x_data = x_data[sorted_indices]
                y_data = y_data[sorted_indices]

                y_data = y_max - (y_data - y_min)

                ax2.plot(x_data, y_data, linewidth=3, color=colors[i])

        i += 1


    plt.figure(figsize=(8, 7))
    ax1 = plt.subplot(212)
    ax1.set_title('Segmented')


    for i in range(n_classes):
        start_idx = predicted_indices[i]
        end_idx = start_idx + len(classes_xyz[i]) # for subtask length
        print(i, start_idx, end_idx)
        ax1.plot(full_x_data[start_idx:end_idx], full_y_data[start_idx:end_idx], 
             linewidth=3, color=colors[i], label=f"Class {i+1}") 

    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.legend()

    '''
    for i in range(n_classes):
        ax1.plot(np.arange(predicted_indices[i], predicted_indices[i]+len(classes_xyz[i])), 
                 full_task_xyz[predicted_indices[i]:predicted_indices[i]+len(classes_xyz[i])], 
                 colors[i])
    '''

    i = 0
    for file in files:
        ax2 = plt.subplot(2, n_classes, i+1)
        ax2.set_title(f'Class {i+1}')
        with h5py.File(f"/Users/wendy/Desktop/school/uml/robotics/dog/h5 files/{file}.h5", 'r') as hf:

            all_y_values = []
            for key in hf["smoothed"].keys():
                y_data = np.array(hf[f"smoothed/{key}/y"])
                all_y_values.extend(y_data)
            
            y_min, y_max = min(all_y_values), max(all_y_values)

            for key in sorted(hf["smoothed"].keys(), key=int): 
                x_data = np.array(hf[f"smoothed/{key}/x"])
                y_data = np.array(hf[f"smoothed/{key}/y"])
                t_data = np.array(hf[f"smoothed/{key}/t"])

                sorted_indices = np.argsort(t_data)
                x_data = x_data[sorted_indices]
                y_data = y_data[sorted_indices]

                y_data = y_max - (y_data - y_min)

                ax2.plot(x_data, y_data, linewidth=3, color=colors[i])

        i += 1

    plt.show()

if __name__ == "__main__":
    main()