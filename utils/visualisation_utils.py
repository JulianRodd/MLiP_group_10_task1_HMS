import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils.general_utils import get_logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.general_utils import get_logger
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils.general_utils import get_logger


def plot_spectrogram(data):
    """
    Visualize spectrogram data.
    
    Args:
    data (np.ndarray): Spectrogram data to visualize.
    """

    logger = get_logger('utils/vizualisation_utils/plot_spectrogram')
    logger.info("Plotting spectrogram with Plotly")

    fig = go.Figure(data=go.Heatmap(
        z=np.log(data + 1e-6), # Log transform with a small constant to avoid log(0)
        colorscale='Viridis'
    ))

    fig.update_layout(
        title='Spectrogram Visualization',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Frequency')
    )

    fig.show()


def plot_eeg_combined_graph(eeg_spectrogram, window_size=100):
    """
    Prints a single line graph for a given eeg_spectrogram, with different lines representing different channels.
    Includes smoothing using a moving average.

    Parameters:
    eeg_spectrogram (np.ndarray): The EEG spectrogram data with shape (128, 256, 4).
    window_size (int): The window size for the moving average smoothing.
    """
    colors = ["blue", "red", "green", "purple"]
    labels = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]

    fig = go.Figure()

    for i in range(4):
        smoothed_data = moving_average(eeg_spectrogram[:, :, i].flatten(), window_size)
        fig.add_trace(
            go.Scatter(
                y=smoothed_data,
                mode="lines",
                name=labels[i],
                line=dict(color=colors[i]),
            )
        )

    fig.update_layout(
        title="EEG Combined Graph", xaxis_title="Time Instances", yaxis_title="Voltage"
    )
    fig.show()


def moving_average(data, window_size = 100):
    """
    Compute the moving average of the given data using a specified window size.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

