import os
from copy import deepcopy

import pandas as pd
import numpy as np
import pickle
from logging import getLogger, basicConfig, INFO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from datasets.raw_data_loader import CustomRawDataset
from generics.configs import DataConfig
from utils.evaluation_utils import score_kl_divergence


os.chdir("/home/janneke/Documents/Master/Machine_Learning_in_Practice/HMS/MLiP_group_10_task1_HMS/")

basicConfig(level=INFO)
logger = getLogger('main')
config = DataConfig()

dataset = CustomRawDataset(config, mode="test", cache=True)
dataset.print_summary()

# dataset.features_per_sample

with open("ensemble_one_model_per_target.pickle", "rb") as pickle_file:
	models = pickle.load(pickle_file)

# y_pred = np.zeros(y_test.shape)
	
# for i, lbl_group in enumerate([y0, y1, y2, y3, y4, y5]):
# 	# y_train_group = lbl_group
# 	# clf = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train, y_train_group)
# 	# clf.fit(x_train, y_train_group)
# 	y_pred_group = clf.predict(x_test)
# 	y_pred[:,i] = y_pred_group
# 	# models.append(clf)

# y_pred[y_pred < 0] = 0
# y_pred_probabilities = y_pred / np.sum(y_pred, axis=1)[:,None]

