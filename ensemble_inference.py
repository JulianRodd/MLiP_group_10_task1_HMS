import pickle
from logging import INFO, basicConfig, getLogger

import numpy as np
import torch

from datasets.data_loader_configs import BaseDataConfig
from datasets.raw_data_loader import CustomRawDataset
from generics import Generics, Paths
from utils.inference_utils import create_submission

basicConfig(level=INFO)
logger = getLogger("main")
config = BaseDataConfig()
paths = Paths()


def main(
    models=None,
    submission_file="submission.csv",
    normalized=True,
    feature_list=["desc"],
    train_subset_sample_count=1000,
):
    test_dataset = CustomRawDataset(
        config, paths, mode="test", cache=False, feature_list=feature_list
    )
    test_dataset.print_summary()

    if not models:
        with open(
            f"{paths.OTHER_MODEL_CHECKPOINTS}ensemble_one_model_per_target_{train_subset_sample_count}_{'norm_' if normalize else ''}feats({'_'.join(sorted(feature_list))}).pickle",
            "rb",
        ) as pickle_file:
            models = pickle.load(pickle_file)

    means = models.pop("means")
    stds = models.pop("stds")
    if normalized:
        test_dataset.features_per_sample = normalize(means, stds, test_dataset)

    x_test = test_dataset.features_per_sample
    y_pred = np.zeros((x_test.shape[0], len(models)))

    print(sum(torch.isinf(torch.Tensor(x_test))))

    for i, (lbl, model) in enumerate(models.items()):
        y_pred_group = model.predict(x_test)
        y_pred[:, i] = y_pred_group

    y_pred[y_pred < 0] = 0
    y_pred_probabilities = y_pred / np.sum(y_pred, axis=1)[:, None]

    create_submission(
        test_dataset.main_df, y_pred_probabilities, Generics.LABEL_COLS, submission_file
    )
    print("Submission created!")


def normalize(means, stds, test_dataset):
    num_channels = 19
    num_features = int(test_dataset.features_per_sample.shape[1] / num_channels)
    one_hot_len = len(test_dataset.config.NAMES)

    normalized_features = np.zeros(test_dataset.features_per_sample.shape)
    for i in range(num_channels):
        features = test_dataset.features_per_sample[
            :, i * num_features : i * num_features + num_features - one_hot_len
        ]

        # Get mean and std from the train set
        mean = means[i * num_features : i * num_features + num_features - one_hot_len]
        std = stds[i * num_features : i * num_features + num_features - one_hot_len]

        normalized_features[
            :, i * num_features : i * num_features + num_features - one_hot_len
        ] = (features - mean) / std

        one_hot = test_dataset.features_per_sample[
            :,
            i * num_features
            + num_features
            - one_hot_len : i * num_features
            + num_features,
        ]
        normalized_features[
            :,
            i * num_features
            + num_features
            - one_hot_len : i * num_features
            + num_features,
        ] = one_hot

    return normalized_features


if __name__ == "__main__":
    train_val_size = 5000
    normalized = True
    feature_list = ["hfda"]
    main(
        normalized=normalized,
        feature_list=feature_list,
        train_subset_sample_count=train_val_size,
    )
