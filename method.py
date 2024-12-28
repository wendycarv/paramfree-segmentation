import numpy as np
import GPy
import matplotlib

# use gpy library for gaussian process
# find an hsmm library? or make it ourselves????

'''
# Step 1: Preprocess Data
# load all subtasks and tasks (xyz .txt files) into arrays
subtasks = load_and_preprocess_subtasks("subtask_files")
larger_task = load_and_preprocess_task("larger_task_file")

# assign truth (y) labels to subtasks which should be easy cause we have a spreadsheet for them (that was created from memory but i think is mostly correct)

# Step 2: Initialize GP-HSMM
# define kernel (GPy.kernel), input data (time or sequence indices), observations (xyz data)
gps = train_gaussian_processes(subtasks)  # train one GP per subtask

# define forward filtering (likelihood of observations up to time t for all possible segment boundaries and classes)
# hsmm also defined here
# args: time-series data (n_samples, n_features), initial_probs (initial probabilities of each class)
# returns the probabilities of each class at each time step...

# define backward sampling (reconstruct most likely sequence of segment boundaries and classes by sampling from posterior distributions)
# args: forward probabilities computed via forward filtering, transition_matrix (transition probabilities between subtasks)
# returns class_sequence (most likely sequence of classes)
^^ we already have this technically, just comparing it

# hsmm components: states (subtasks), durations (expected for each task), gps emissions (use gps to model the outputs)
hsmm = initialize_hsmm(gps, duration_models, transition_probs)
# Step 3: Train HSMM
hsmm.train(training_data=subtasks)

# Step 4: Segment and Identify Subtasks
segmentation = hsmm.infer(larger_task)

# Step 5: Visualize Results
visualize_segmentation(larger_task, segmentation)

# Step 6: Evaluate
evaluate_segmentation(segmentation, ground_truth)

'''