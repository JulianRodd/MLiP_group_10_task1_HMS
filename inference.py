import numpy as np
import pickle
from logging import getLogger, basicConfig, INFO

from generics import Generics, Paths
from datasets.raw_data_loader import CustomRawDataset
from datasets.data_loader_configs import BaseDataConfig
from utils.inference_utils import create_submission


basicConfig(level=INFO)
logger = getLogger('main')
config = BaseDataConfig()
paths = Paths()

def main(models=None, submission_file='submission.csv'):
	train_subset_sample_count = 0
	test_dataset = CustomRawDataset(config, paths, mode="test", cache=True)
	test_dataset.print_summary()

	x_test = test_dataset.features_per_sample
	print(x_test.shape)

	if not models:
		with open(f"{paths.OTHER_MODEL_CHECKPOINTS}ensemble_one_model_per_target_{train_subset_sample_count}.pickle", "rb") as pickle_file:
			models = pickle.load(pickle_file)

	y_pred = np.zeros((x_test.shape[0], len(models)))  # shape: num_samles, num_labels

	for i, (lbl, model) in enumerate(models.items()):
		y_pred_group = model.predict(x_test)
		y_pred[:,i] = y_pred_group

	y_pred[y_pred < 0] = 0
	y_pred_probabilities = y_pred / np.sum(y_pred, axis=1)[:,None]

	submission_df = create_submission(test_dataset.main_df, y_pred_probabilities, Generics.LABEL_COLS, submission_file)