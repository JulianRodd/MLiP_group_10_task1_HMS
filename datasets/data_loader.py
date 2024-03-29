import os
import gc
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import albumentations as A
from generics import Paths, Generics
from prettytable import PrettyTable
from utils.general_utils import get_logger
from utils.loader_utils import (
    load_eeg_spectrograms,
    load_spectrograms,
    load_preloaded_eeg_spectrograms,
    load_preloaded_spectrograms,
    normalize_eeg_spectrograms,
)
from utils.visualisation_utils import plot_eeg_combined_graph, plot_spectrogram
from torch.utils.data import DataLoader, Dataset
from utils.ica_utils import apply_ica_to_eeg_spectrograms
from utils.mspca_utils import apply_mspca_to_eeg_spectrograms


class CustomDataset(Dataset):
    """
    Custom Dataset for EEG data.

    Attributes:
        config (DataConfig): Configuration object containing dataset parameters.
        augment (bool): Flag to determine if augmentation should be applied.
        mode (str): Operating mode ('train' or 'test').
        spectrograms (dict): Dictionary to hold EEG spectrograms.
        cache (bool): Flag to determine if the dataset should be cached.
        tensorboard_prefix (str): Prefix for the TensorBoard logs.
    """

    def __init__(
        self,
        config,
        main_df: pd.DataFrame,
        augment: bool = False,
        mode: str = "train",
        cache: bool = True,
        tensorboard_prefix: str = "all",
    ):
        self.logger = get_logger("data_loader.log")
        self.config = config
        self.main_df = main_df
        self.label_cols = Generics.LABEL_COLS
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                Paths.TENSORBOARD_DATASETS, f"{tensorboard_prefix}/{config.NAME}_{mode}"
            )
        )
        self.augment = augment
        self.mode = mode
        self.spectrograms = {}
        self.eeg_spectrograms = {}

        if mode == "test":
            self.batch_size = config.BATCH_SIZE_TEST
        elif mode == "val":
            self.batch_size = config.BATCH_SIZE_VAL
        else:
            self.batch_size = config.BATCH_SIZE_TRAIN

        cache_file = self.generate_cache_filename(self.config.SUBSET_SAMPLE_COUNT, mode)
        if os.path.exists(f"{Paths.CACHE_PATH_READ}{cache_file}") and cache:
            self.logger.info(f"Loading dataset from cache: {cache_file}")
            self.load_from_cache(cache_file)
        else:
            self.logger.info("Processing and caching new dataset")
            if self.config.USE_PRELOADED_EEG_SPECTROGRAMS:
                self.eeg_spectrograms = load_preloaded_eeg_spectrograms(
                    self.main_df, custom_config=self.config.PREPROCESSING
                )
            else:
                self.eeg_spectrograms = load_eeg_spectrograms(
                    main_df=self.main_df,
                    mode=self.mode,
                    feats=self.config.FEATS,
                    use_wavelet=self.config.USE_WAVELET,
                    mspca_on_raw_eeg=self.config.APPLY_MSPCA_RAW_EEG,
                    ica_on_raw_eeg=self.config.APPLY_ICA_RAW_EEG,
                    custom_config=self.config.PREPROCESSING,
                )

            if self.config.NORMALIZE_EEG_SPECTROGRAMS:
                self.eeg_spectrograms = normalize_eeg_spectrograms(
                    self.eeg_spectrograms, self.config.NORMALIZE_INDIVIDUALLY
                )

            if self.config.APPLY_ICA_EEG_SPECTROGRAMS:
                self.eeg_spectrograms = apply_ica_to_eeg_spectrograms(
                    self.eeg_spectrograms
                )
                if self.config.NORMALIZE_EEG_SPECTROGRAMS:
                    self.eeg_spectrograms = normalize_eeg_spectrograms(
                        self.eeg_spectrograms, self.config.NORMALIZE_INDIVIDUALLY
                    )

            if self.config.APPLY_MSPCA_EEG_SPECTROGRAMS:
                self.eeg_spectrograms = apply_mspca_to_eeg_spectrograms(
                    self.eeg_spectrograms, n_components=self.config.N_COMPONENTS
                )
                if self.config.NORMALIZE_EEG_SPECTROGRAMS:
                    self.eeg_spectrograms = normalize_eeg_spectrograms(
                        self.eeg_spectrograms
                    )

            if self.config.USE_PRELOADED_SPECTROGRAMS:
                self.spectrograms = load_preloaded_spectrograms(self.main_df)
            else:
                self.spectrograms = load_spectrograms(
                    main_df=self.main_df, mode=self.mode
                )

            if cache:
                self.cache_data(cache_file)

        self.logger.info(
            f"Dataset loaded: {self.mode} mode, {len(self.main_df)} samples, with config {self.config.NAME}"
        )

    def generate_cache_filename(self, subset_sample_count: int, mode: str) -> str:
        """
        Generates a unique filename for caching the dataset based on the provided configuration.

        Args:
            subset_sample_count (int): Number of samples to subset.
            mode (str): Dataset mode ('train' or 'test').

        Returns:
            str: Filename for caching the dataset.
        """
        config_summary = f"{self.config.NAME}_{subset_sample_count}_{mode}"
        return f"{config_summary}.npz"

    def cache_data(self, cache_file: str):
        """
        Caches the dataset to a file.

        Args:
            cache_file (str): The file path where the dataset will be cached.
        """
        np.savez(
            f"{Paths.CACHE_PATH_WRITE}{cache_file}",
            main_df=self.main_df.to_records(index=False),
            spectrograms=self.spectrograms,
            eeg_spectrograms=self.eeg_spectrograms,
        )
        self.logger.info(f"Dataset cached at {cache_file}")

    def load_from_cache(self, cache_file: str):
        """
        Loads the dataset from the cache file.

        Args:
            cache_file (str): The file path from which the dataset will be loaded.
        """
        cached_data = np.load(f"{Paths.CACHE_PATH_READ}{cache_file}", allow_pickle=True)
        self.spectrograms = cached_data["spectrograms"].item()
        self.eeg_spectrograms = cached_data["eeg_spectrograms"].item()
        self.label_cols = Generics.LABEL_COLS

    def get_torch_data_loader(self):
        """
        Get the torch data loader for the dataset.

        Args:
            config (DataConfig): Configuration for the dataset.

        Returns:
            DataLoader: Data loader for the dataset.
        """
        try:
            return DataLoader(
                self,
                batch_size=self.batch_size,
                shuffle=self.config.SHUFFLE_TRAIN,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY,
                drop_last=self.config.DROP_LAST,
            )
        except Exception as e:
            self.logger.error(f"Error getting data loader: {e}")
            raise

    def __len__(self) -> int:
        """
        Get the total number of items in the dataset.

        Returns:
            int: Total number of items.
        """
        return len(self.main_df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the data and its corresponding label.
        """
        try:
            X, y = self.generate_data(index)
            if self.augment:
                X = self.apply_transform(X)
            return torch.tensor(X, dtype=torch.float32), torch.tensor(
                y, dtype=torch.float32
            )
        except Exception as e:
            self.logger.error(f"Error getting item at index {index}: {e}")
            raise

    def generate_data(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a single batch of data.

        This method processes the EEG data to create input features (X) and targets (y)
        based on the specified index in the dataset.

        Args:
            index (int): Index of the batch in the dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the input features and the target values.

        Raises:
            IndexError: If the index is out of the range of the dataset.
            ValueError: If there are issues in processing the data.
        """
        try:
            if index >= len(self.main_df):
                raise IndexError(f"Index {index} is out of range for the dataset.")

            X = np.zeros((128, 256, 8), dtype="float32")
            y = np.zeros(6, dtype="float32")
            row = self.main_df.iloc[index]

            if "min" in row and "max" in row:
                r = int((row["min"] + row["max"]) // 4)
            else:
                r = 0

            for region in range(4):
                img = self.spectrograms[row.spectrogram_id][
                    r : r + 300, region * 100 : (region + 1) * 100
                ].T

                # Processing steps: Log transformation and standardization
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                mu = np.nanmean(img.flatten())
                std = np.nanstd(img.flatten())
                img = (img - mu) / (std + 1e-6)
                img = np.nan_to_num(img, nan=0.0)

                X[14:-14, :, region] = img[:, 22:-22] / 2.0

                X[:, :, 4:] = self.eeg_spectrograms[row.eeg_id]

                if self.mode != "test":
                    y = row[self.label_cols].values.astype(np.float32)

            return X, y

        except IndexError as e:
            self.logger.error(f"Data generation error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in data generation: {e}")
            raise ValueError(f"Error processing data at index {index}")

    def apply_transform(self, img: np.ndarray) -> np.ndarray:
        """
        Apply transformations to the image.

        Args:
            img (np.ndarray): Image to transform.

        Returns:
            np.ndarray: Transformed image.
        """
        transform = A.Compose([A.HorizontalFlip(p=0.5)])
        return transform(image=img)["image"]

    def print_summary(self):
        """
        Prints and logs a summary of the dataset including dataset size, mode, data distribution,
        and vote statistics if in 'train' mode, to both the console and TensorBoard.
        """
        try:
            total_samples = len(self.main_df)
            unique_patients = self.main_df["patient_id"].nunique()
            unique_eegs = self.main_df["eeg_id"].nunique()
            unique_spectrograms = self.main_df["spectrogram_id"].nunique()

            summary_str = f"Dataset Summary:\n"
            summary_str += f"Mode: {self.mode}\n"
            summary_str += f"Total Samples: {total_samples}\n"
            summary_str += f"Unique Patients: {unique_patients}\n"
            summary_str += f"Unique EEGs: {unique_eegs}\n"
            summary_str += f"Unique Spectrograms: {unique_spectrograms}\n"

            self.writer.add_text("Dataset/Summary", summary_str, 0)

            if self.mode == "train":
                augmentation_status = "Enabled" if self.augment else "Disabled"
                summary_str += f"Augmentation: {augmentation_status}\n"

                label_distribution = self.main_df[self.label_cols].sum()
                summary_str += f"Label Distribution:\n{label_distribution}\n"

                # Convert pandas Series to numpy array before logging to TensorBoard
                label_distribution_np = label_distribution.values
                self.writer.add_histogram(
                    "Dataset/Label_Distribution", label_distribution_np, 0
                )

                vote_cols = [
                    "seizure_vote",
                    "lpd_vote",
                    "gpd_vote",
                    "lrda_vote",
                    "grda_vote",
                    "other_vote",
                ]
                vote_stats = self.main_df[vote_cols].agg(["mean", "var"])
                summary_str += f"\nVote Statistics:\n{vote_stats}\n"
                for col in vote_cols:
                    self.writer.add_scalars(
                        f"Dataset/Vote_Stats/{col}",
                        {
                            "mean": vote_stats.loc["mean", col],
                            "var": vote_stats.loc["var", col],
                        },
                        0,
                    )

            summary_str += f"Spectrograms Loaded: {len(self.spectrograms)}\n"
            summary_str += f"EEG Spectrograms Loaded: {len(self.eeg_spectrograms)}\n"

            # Configuration summary
            config_table = PrettyTable()
            config_table.field_names = ["Configuration", "Value"]
            config_table.align = "l"
            for attr in dir(self.config):
                if not attr.startswith("__") and not callable(
                    getattr(self.config, attr)
                ):
                    value = getattr(self.config, attr)
                    config_table.add_row([attr, value])
                    self.writer.add_text(f"Configuration/{attr}", str(value), 0)

            summary_str += "\nConfiguration Summary:\n"
            summary_str += config_table.get_string()

            print(summary_str)
            self.writer.add_text(
                "Dataset/Configuration_Summary", config_table.get_html_string(), 0
            )

        except Exception as e:
            self.logger.error(f"Error printing and logging dataset summary: {e}")
            raise

    def plot_samples(self, n_samples: int = 2):
        """
        Plots n_samples samples each from eeg_spectrograms and spectrograms with corresponding scores.
        """
        try:
            # Plot 2 samples from eegs
            eeg_sample_ids = random.sample(
                list(self.eeg_spectrograms.keys()), n_samples
            )
            for eeg_id in eeg_sample_ids:
                plot_eeg_combined_graph(self.eeg_spectrograms[eeg_id])

            # Plot 2 samples from spectrogram_eegs
            spectrogram_sample_ids = random.sample(
                list(self.spectrograms.keys()), n_samples
            )
            for spectrogram_id in spectrogram_sample_ids:
                plot_spectrogram(self.spectrograms[spectrogram_id])

        except Exception as e:
            self.logger.error(f"Error plotting samples: {e}")
            raise
