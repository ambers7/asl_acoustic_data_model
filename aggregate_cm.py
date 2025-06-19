# aggregate confusion matrices from all runs

import pickle
from acoustic_model_resnet import save_cm_figure

all_ground_truth = []
all_predictions = []

for run in range(1, 6): 
    with open(f'ground_truth_run{run}.pkl', 'rb') as f:
        all_ground_truth.extend(pickle.load(f))
    with open(f'predictions_run{run}.pkl', 'rb') as f:
        all_predictions.extend(pickle.load(f))

save_cm_figure(all_ground_truth, all_predictions, classes, 'cms/acoustic_cnn_cm_all_runs.png')
