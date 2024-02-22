import logging
import os
import torch
import numpy as np
import pandas as pd
import time
from datasets.data_loader import CustomDataset
from generics import Generics, Paths
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(train_dataset, val_dataset, model, tensorboard_prefix: str = "all"):
    """
    Perform cross-validation training and validation.

    Args:
        train_dataset (CustomDataset): The training dataset.
        val_dataset (CustomDataset): The validation dataset.
        model (torch.nn.Module): The model to train.
    """
    oof_df = pd.DataFrame()
    preds, actual = train_loop(train_dataset, val_dataset, model, tensorboard_prefix)

    # Check and reshape preds if necessary
    if len(preds.shape) == 1 or preds.shape[1] == 1:
        preds = preds.reshape(-1, len(Generics.LABEL_COLS))

    preds_df = pd.DataFrame(preds, columns=Generics.LABEL_COLS)
    actual_df = pd.DataFrame(actual, columns=Generics.TARGET_PREDS)

    _oof_df = pd.concat([actual_df, preds_df], axis=1)
    oof_df = pd.concat([oof_df, _oof_df]).reset_index(drop=True)
  
    return oof_df


def train_loop(
    train_dataset: CustomDataset, val_dataset: CustomDataset, model, tensorboard_prefix: str = "all"
):
    preds = []
    actual = []
    total_epochs = model.config.EPOCHS

    tb_run_path = os.path.join(
        Paths.TENSORBOARD_TRAINING,
        f"{tensorboard_prefix}/{train_dataset.config.NAME}_{model.config.NAME}/{train_dataset.config.NAME}_{model.config.NAME}",
    )
    writer = SummaryWriter(tb_run_path)
    try:
        best_loss = np.inf
        model_config = model.config
        criterion = nn.KLDivLoss(reduction=model_config.KLDIV_REDUCTION)
        optimizer = _configure_optimizer(model, model_config)
        scheduler = _configure_scheduler(optimizer, model_config)

        train_loader = train_dataset.get_torch_data_loader()
        val_loader = val_dataset.get_torch_data_loader()

        for epoch in range(model_config.EPOCHS):
            logger.info(f"Epoch {epoch + 1}/{total_epochs}")
            avg_train_loss = _train_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                model.device,
                writer
            )
            avg_val_loss = _valid_epoch(val_loader, model, criterion, model.device, writer, epoch)
            _log_epoch_results(
                epoch,
                avg_train_loss,
                avg_val_loss,
                model,
                train_dataset.config.NAME,
            )

            # Save the best model with fold information
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_name = f"best_{model_config.MODEL}_{model_config.NAME}_{train_dataset.config.NAME}.pth"
                torch.save(
                    model.state_dict(),
                    os.path.join(Paths.BEST_MODEL_CHECKPOINTS, best_model_name),
                )

        # Collect and return final predictions
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            preds.extend(probabilities.cpu().detach().numpy())
            actual.extend(targets.cpu().detach().numpy())
            
        
        writer.close()
        preds = np.array(preds)
        actual = np.array(actual)
        return preds, actual

    except Exception as e:
        logger.error(f"An error occurred in train_loop: {e}")
        writer.close()
        raise


def _configure_optimizer(model, config):
    """
    Configures the optimizer for the model based on the provided configuration.

    Args:
        model (torch.nn.Module): The model for which the optimizer will be configured.
        config (object): A configuration object containing optimizer settings.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    optimizer_type = config.OPTIMIZER.lower()

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.OPTIMIZER}")

    return optimizer


def _configure_scheduler(optimizer, config):
    """
    Configures the learning rate scheduler for the optimizer based on the provided configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the scheduler will be configured.
        config (object): A configuration object containing scheduler settings.

    Returns:
        torch.optim.lr_scheduler: Configured learning rate scheduler.
    """
    scheduler_type = config.SCHEDULER.lower()

    if scheduler_type == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
        )
    elif scheduler_type == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {config.SCHEDULER}")

    return scheduler


def _train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device, writer):
    """
    Handles the training of the model for one epoch.

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The neural network model to train.
        criterion (torch.nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        epoch (int): Current epoch number.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Device on which to perform computations.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0
    total_batches = len(train_loader)
    i = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Reset gradients to zero

        # Forward pass
        with torch.cuda.amp.autocast(enabled=model.config.AMP):
            outputs = model(inputs)
            outputs = torch.nn.functional.log_softmax(
                outputs, dim=1
            )  # Ensure log probabilities
            loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update learning rate
        scheduler.step()
        writer.add_scalar("Loss/Train", loss.item(), i + total_batches*epoch)
        
        i += 1

    average_loss = total_loss / total_batches
    return average_loss


def _valid_epoch(val_loader, model, criterion, device, writer, epoch=0):
    """
    Handles the validation of the model for one epoch.

    Args:
        val_loader (DataLoader): DataLoader for the validation data.
        model (torch.nn.Module): The neural network model to validate.
        criterion (torch.nn.Module): Loss function used for validation.
        device (torch.device): Device on which to perform computations.

    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_batches = len(val_loader)
    i = 0
    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            outputs = torch.nn.functional.log_softmax(outputs + 1e-6, dim=1)
            loss = criterion(outputs, targets)

            if loss.item() < 0:  # Add a check for negative loss
              logger.warning(f"Negative loss encountered: {loss.item()}")
              logger.warning(f"Outputs: {outputs}")
              logger.warning(f"Targets: {targets}")
            
            total_loss += loss.item()
            writer.add_scalar("Loss/Validation", loss.item(), i + total_batches*epoch)
            
            i += 1
            # Additional operations for metrics or logging can be added here

    average_loss = total_loss / total_batches
    return average_loss


def _log_epoch_results(epoch, avg_train_loss, avg_val_loss, model, dataset_name):
    """
    Logs the results at the end of each epoch during training and validation.

    Args:
        epoch (int): The current epoch number.
        avg_train_loss (float): The average training loss for the epoch.
        avg_val_loss (float): The average validation loss for the epoch.
        prediction_dict (dict): A dictionary containing predictions and other relevant information.
        model (torch.nn.Module): The model being trained and validated.
        dataset_name (str): The name of the dataset used for training and validation.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
    logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")


def _collect_final_predictions(val_loader, model, device):
    """
    Collects the final predictions from the model on the validation dataset.

    Args:
        val_loader (DataLoader): DataLoader for the validation data.
        model (torch.nn.Module): The trained neural network model.
        device (torch.device): Device on which to perform computations.

    Returns:
        List: A list of model predictions.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, _ in val_loader:
            inputs = inputs.to(device)

            # Forward pass to get outputs/predictions
            outputs = model(inputs)
            outputs = torch.nn.functional.log_softmax(outputs + 1e-6, dim=1)
            # Convert outputs to probabilities and then to CPU for further processing if needed
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions.extend(probabilities.cpu().numpy())

    return predictions


def get_result(df, label_cols = Generics.LABEL_COLS, target_preds = Generics.TARGET_PREDS):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(df[label_cols].values)
    preds = torch.tensor(df[target_preds].values)
    preds = F.log_softmax(preds, dim=1)
    result = kl_loss(preds, labels)
    return result
