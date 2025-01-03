import numpy as np
import GPy
import matplotlib.pyplot as plt
import os
import glob
from hmmlearn.hmm import GaussianHMM
from scipy.stats import poisson, norm

# TODO: UNDERSTAND, implement GPs
class GP_HSMM:
    def __init__(self, num_states, duration_distributions, gp_emissions, transition_matrix):
        """
        :param num_states: # of HSMM states (subtasks)
        :param duration_distributions: list of duration distributions for each state
        :param gp_emissions: list of trained GP models (one for each task)
        :param transition_matrix: state transition probabilities
        """
        self.num_states = num_states
        self.duration_distributions = duration_distributions
        self.gp_emissions = gp_emissions
        self.transition_matrix = transition_matrix

    def forward_filtering(self, observations):
        """
        compute likelihoods of each state at each time step
        :param observations:
        :return:
        """
        num_timesteps = len(observations)
        alpha = np.zeros((num_timesteps, self.num_states))

        # initialize the first step
        for state in range(self.num_states):
            emission_prob = self.compute_emission_prob(state, observations[0])
            duration_prob = self.duration_distributions[state].pmf(1)  # Duration of 1 timestep
            alpha[0, state] = emission_prob * duration_prob

        # Recursive calculation
        for t in range(1, num_timesteps):
            for state in range(self.num_states):
                emission_prob = self.compute_emission_prob(state, observations[t])
                duration_prob = self.duration_distributions[state].pmf(t)  # Adjust duration logic as needed
                alpha[t, state] = np.sum(
                    alpha[t - 1, :] * self.transition_matrix[:, state]) * emission_prob * duration_prob

        return alpha

    def backward_sampling(self, observations, alpha):
        """
        perform backward sampling to determine the most likely state sequence.
        """
        num_timesteps = len(observations)
        state_sequence = np.zeros(num_timesteps, dtype=int)

        # start from the last time step
        state_sequence[-1] = np.argmax(alpha[-1, :])

        # backtrack
        for t in range(num_timesteps - 2, -1, -1):
            prev_state = state_sequence[t + 1]
            state_sequence[t] = np.argmax(alpha[t, :] * self.transition_matrix[:, prev_state])

        # get subtask boundaries
        boundaries = []
        for t in range(1, num_timesteps):
            if state_sequence[t] != state_sequence[t-1]:  # we've found a state change
                boundaries.append((t-1, t))  # (start, end) of the subtask

        return state_sequence, boundaries

    def compute_emission_prob(self, state, observation):
        """
        Compute the likelihood of an observation given a state using the state's GP model.
        """
        gp_model = self.gp_emissions[state]
        mu, sigma = gp_model.predict([observation])
        return norm.pdf(observation, loc=mu, scale=np.sqrt(sigma))

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
        for file in subtask_files:
            #print(f"Loading {file} for subtask {subtask_name}")
            data = np.loadtxt(file)  # load file into an array
            #print(data.shape)
            #subtasks_data[subtask_name].append(data)  # append to subtask dictionary
            #print(subtasks_data[subtask_name])
            if subtask_name in subtasks_data:
                subtasks_data[subtask_name] = np.vstack((subtasks_data[subtask_name], data))
            else:
                subtasks_data[subtask_name] = data

    # convert dictionary of arrays of individual subtasks  into one long array of the same subtask.....

    # search for full task files
    full_tasks_files = glob.glob(os.path.join(path, 'full_tasks', '*.txt'))
    for file in full_tasks_files:
        #print(f"Loading {file} for full task")
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
    # step 3: initialize HSMM
    # components: states (subtasks), durations (expected for each task), gps emissions (use gps to model the outputs)
    '''
    num_subtasks = 5  # plate, napkin, cup, spoon, fork
    # transition_matrix - order of subtasks currently always deterministically 1, 2, 3, 4, 5
    transition_matrix = np.zeros((num_subtasks, num_subtasks))
    for i in range(num_subtasks - 1):
        transition_matrix[i, i + 1] = 1  # subtask i transitions to i + 1

    hsmm = GP_HSMM(num_states=num_subtasks, duration_distributions=duration_distribution, gp_emissions=gp_models, transition_matrix=transition_matrix)

    '''
    # step 4: segment and identify subtasks
    '''
    boundaries = []
    for task_data in full_tasks_data:
        print("Processing full task")
        alpha = hsmm.forward_filtering(task_data)
        state_sequence, boundaries = hsmm.backward_sampling(task_data, alpha)  # grab boundaries/segmentation here?
        print("Predicted state sequence: ", state_sequence)
        print("Subtask boundaries: ", boundaries)
        '''
        # step 5: visualize segments using matplot
        '''
        visualize_segmentation(task_data, boundaries)

'''
# step 6: evaluate
evaluate_segmentation(segmentation, ground_truth)
'''

