import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dtaidistance import dtw_ndim
from dtaidistance import dtw_ndim_visualisation as dtwvis
from scipy.interpolate import interp1d

from tslearn.metrics import dtw_path_from_metric

# use each subtask and compare with larger subtask

# import xyz data for subtask and for larger task
def load_xyz_data(file):
    data = np.loadtxt(file)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    task = np.array([x, y, z])
    #task = np.array([x, y])

    return task


def main():
    subtask = load_xyz_data("/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/xyz data/fetch_recorded_demo_1730996265.txt")
    maintask = load_xyz_data("/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/xyz data/fetch_recorded_demo_1730997956.txt")

    # Compute DTW path and distance
    #path, distance = dtw_path_from_metric(subtask, maintask, metric="euclidean")

    # Assume 'subtask' and 'main_task' are arrays with shape (n_samples, 3), where n_samples is the length of the task
    subtask_length = subtask.shape[1]
    main_task_length = maintask.shape[1]

    # Interpolate the subtask to match the main task length
    interpolated_subtask = np.zeros_like(maintask)

    for i in range(3):  # For each feature (x, y, z)
        # Create an interpolator for the i-th feature
        interpolator = interp1d(np.linspace(0, 1, subtask_length), subtask[i, :], kind='cubic', axis=0)
        # Apply the interpolator to get the interpolated values at the new time steps
        interpolated_subtask[i, :] = interpolator(np.linspace(0, 1, main_task_length))

    print(interpolated_subtask)
    # Now 'interpolated_subtask' will have the same shape as the main task, (3, 50)
    print(interpolated_subtask.shape)

    distance = dtw_ndim.distance(interpolated_subtask, maintask)
    print(distance)
    path = dtw_ndim.warping_path(interpolated_subtask, maintask)

    # Find the segment of the main task that matches the subtask
    path_indices = np.array(path)
    start_idx = path_indices[0][1]  # First index of the path in the main task
    end_idx = path_indices[-1][1]  # Last index of the path in the main task

    # Extract the matching segment from the main task
    matching_segment = maintask[:, start_idx:end_idx + 1]

    # Plot the subtask and the corresponding segment from the main task
    plt.figure(figsize=(10, 6))

    # Plot subtask
    plt.subplot(3, 1, 1)
    plt.plot(subtask[0, :], subtask[1, :], subtask[2, :], label="Subtask", color="blue")
    plt.title("Subtask (x, y, z)")

    plt.subplot(3, 1, 2)
    plt.plot(maintask[0, :], maintask[1, :], maintask[2, :], label="Main task", color="green")
    plt.title("Main task (x, y, z)")

    # Plot matching segment from the main task
    plt.subplot(3, 1, 3)
    plt.plot(matching_segment[0, :], matching_segment[1, :], matching_segment[2, :], label="Matching Segment", color="red")
    plt.title(f"Matching Segment from Main Task (start: {start_idx}, end: {end_idx})")

    # Show the plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# should return indexes of time series that match smaller subtask
# with that, should accumulate all indexes that match and plot the larger task with vertical dividers
# note that there should be three separate plots for x y z, not a 3d graph

# apply DTW on the discrete derivative of each curve using numpy.diff() to better capture the curves’ dynamic or shape.




# if that does not work, try GP-HSMM...


