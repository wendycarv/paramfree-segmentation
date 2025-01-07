import numpy as np
import GPy
import matplotlib.pyplot as plt
import os
import glob
from hmmlearn.hmm import GaussianHMM
from scipy.stats import poisson, norm

'''
# step 1: preprocess data
# assign truth (y) labels to subtasks which should be easy cause we have a spreadsheet for them (that was created from memory but i think is mostly correct)
'''
def load_data_from_path(path):
    subtasks_data = {}
    full_tasks_data = []

    # search for subtask files grouped by folder
    subtasks_folders = glob.glob(os.path.join(path, 'subtasks', '*'))  # get subfolders
    for folder in subtasks_folders:
        subtask_name = os.path.basename(folder)  # get subtask folder name (1-5)

        # get all .txt files within subfolder
        subtask_files = glob.glob(os.path.join(folder, '*.txt'))

        if subtask_name not in subtasks_data:
            subtasks_data[subtask_name] = []

        for file in subtask_files:
            data = np.loadtxt(file)  # load file into an array

            if subtask_name in subtasks_data:
                #subtasks_data[subtask_name] = np.vstack((subtasks_data[subtask_name], data))
                subtasks_data[subtask_name].append(data)  # append to subtask dictionary
            else:
                subtasks_data[subtask_name] = data

    # convert dictionary of arrays of individual subtasks  into one long array of the same subtask.....

    # search for full task files
    full_tasks_files = glob.glob(os.path.join(path, 'full_tasks', '*.txt'))
    for file in full_tasks_files:
        data = np.loadtxt(file)  # load the file into an array
        full_tasks_data.append(data) # add array to list

    return subtasks_data, full_tasks_data


'''
# step 2: initialize GP-HSMM
# define kernel (GPy.kernel), input data (time or sequence indices), observations (xyz data)
gps = train_gaussian_processes(subtasks)  # train one GP per subtask
'''
def train_gp_for_subtask(data):
    # data has the shape (time_steps, 3)
    time_steps = np.arange(len(data)).reshape(-1, 1)
    print(f"Time steps: {time_steps.shape[0]}")
    models = []

    # train GP for each dimension (x, y, z)
    for dim in range(data.shape[1]):
        kernel = GPy.kern.RBF(input_dim=1)
        model = GPy.models.GPRegression(time_steps, data[:, dim:dim+1], kernel)
        model.optimize(messages=True)
        models.append(model)

    return models  # list of trained GP models (one per dimension)


def visualize_segmentation(data, boundaries):
    time_steps = np.arange(len(data))
    plt.plot(time_steps, data[:, 0], label="X")
    plt.plot(time_steps, data[:, 1], label="Y")
    plt.plot(time_steps, data[:, 2], label="Z")

    # vertical lines for segment boundaries
    for start, end in boundaries:
        plt.axvline(x=start, color="r", linestyle="--")
        plt.axvline(x=end, color="g", linestyle="--")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    path = '/Users/wendy/Desktop/school/uml/robotics/fetch_table_demos/xyz data/'
    subtasks_data, full_tasks_data = load_data_from_path(path)

    '''
    gp_models = {}
    prev_label = ""
    duration_distribution = {}
    durations = []

    # train one GP for each subtask and grab duration distribution
    for subtask, data, in list(subtasks_data.items()):
        current_label = subtask
        print("Subtask ", subtask, ": ", data.shape)
        durations.append(data.shape[0])  # add duration of current subtask
        gp_models[subtask] = train_gp_for_subtask(data)
        if current_label != prev_label and prev_label != "":  # if label changes, calculate lambda of previous subtask
            mean = np.mean(durations)
            duration_distribution[prev_label] = poisson(mu=mean)
            print(f"Subtask {prev_label}, distribution: {duration_distribution[prev_label]}")
            durations = []  # reset durations for the next subtask

        prev_label = current_label

    # handling last subtask
    if durations:
        lambda_value = np.mean(durations)
        duration_distribution[prev_label] = {lambda_value}
        print(f"Subtask {prev_label}, mean distribution: {lambda_value}")

    print(f"Duration Distributions: {duration_distribution}")
    print(f"Trained {len(gp_models)} GP models for subtasks: {list(gp_models.keys())}")
    '''

    '''
    # step 3: initialize HSMM
    # components: states (subtasks), durations (expected for each task), gps emissions (use gps to model the outputs)
    '''
    num_subtasks = 5  # plate, napkin, cup, spoon, fork
    # transition_matrix - order of subtasks currently always deterministically 1, 2, 3, 4, 5
    transition_matrix = np.zeros((num_subtasks, num_subtasks))
    for i in range(num_subtasks - 1):
        transition_matrix[i, i + 1] = 1  # subtask i transitions to i + 1

    # Normalize rows to sum to 1, but skip the last row
    for i in range(num_subtasks - 1):  # Exclude last row (subtask 5)
        transition_matrix[i] /= transition_matrix[i].sum()

    transition_matrix[4][4] = 1

    plate_sequences = subtasks_data["1"]
    napkin_sequences = subtasks_data["2"]
    cup_sequences = subtasks_data["3"]
    fork_sequences = subtasks_data["4"]
    spoon_sequences = subtasks_data["5"]

    all_seq = plate_sequences + napkin_sequences + cup_sequences + fork_sequences + spoon_sequences  # all examples from each subtask
    one_each_seq = np.concatenate([plate_sequences[0], napkin_sequences[0], cup_sequences[0], spoon_sequences[0]])  # only 1 example from each
    concat_data = np.vstack(all_seq)

    all_seq_lengths = [len(seq) for seq in all_seq]
    print(all_seq_lengths)
    one_seq_lengths = [len(plate_sequences[0]), len(napkin_sequences[0]), len(cup_sequences[0]), len(spoon_sequences[0])]
    print(one_seq_lengths)
    
    # number of hidden states = 5; covariance type (covariance matrix) = diag, full?; trans_mat=transition_matrix* (maybe not needed rn?), n_iter can vary...
    hmm_model = GaussianHMM(n_components=num_subtasks, covariance_type="diag", n_iter=100, algorithm='viterbi', init_params='c')
    hmm_model.transmat_ = transition_matrix

    # (X, lengths): concat_data = np.concatenate([plate, napkin, cup, fork, spoon]), lengths = [len(plate), len(napkin),...] (len(task)=# of examples we have for that subtask)
    print("Training model...")
    hmm_model.fit(concat_data, all_seq_lengths)
    print("Finished training model")

    print("Predicting task...")
    pred_states = hmm_model.predict(full_tasks_data[0])
    print(pred_states)

    # need to find actual boundaries of these states too
    boundaries = []
    # Iterate through the predicted states to find transitions
    for i in range(1, len(pred_states)):
        if pred_states[i] != pred_states[i - 1]:
            boundaries.append(i)

    # Print or use the boundaries
    print("State boundaries:", boundaries)

    states = []
    start = 0
    # Output the subtasks and their boundaries
    for i in range(len(pred_states) - 1):
        if pred_states[i] != pred_states[i+1]:
            end = i
            states.append((pred_states[start], start, end))
            start = i + 1

    states.append((pred_states[start], start, len(pred_states) - 1))
    
    for state, start, end in states:
        print(f"Subtask: {state}, Start: {start}, End: {end}")