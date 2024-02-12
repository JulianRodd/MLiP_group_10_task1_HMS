import math
import os
import random
import time
import numpy as np
import os
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
import torch
from generics import Paths


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s: float):
    "Convert to minutes."
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since: float, percent: float):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


import os
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

def get_logger(filename):
    """
    Creates a logger with both stream and file handlers.

    Args:
    filename (str): Path and name of the log file.

    Returns:
    logger: Configured logger object.
    """
    logger = getLogger(__name__)
    logger.setLevel(INFO)

    # Stream Handler for console output
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler1)

    # File Handler for file output
    log_file_path = Paths.LOG_PATH + filename + ".log"

    # Create directories if they don't exist (only if a directory is specified in the path)
    dir_name = os.path.dirname(log_file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    handler2 = FileHandler(log_file_path)
    handler2.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler2)

    return logger



def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sep():
    print("-" * 100)
