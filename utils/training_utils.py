
import os
import torch
import numpy as np
import pandas as pd
import time
from scipy.special import kl_div
from datasets.data_loader import CustomDataset
from generics import Generics, Paths
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.general_utils import AverageMeter, get_logger, timeSince
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import mean_squared_error
# Configure logger
logger = get_logger(__name__)

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
        train_loader = train_dataset.get_torch_data_loader()
        scheduler = OneCycleLR(
            optimizer,
            max_lr=model_config.MAX_LEARNING_RATE_SCHEDULERER,
            epochs=model_config.EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.05,
            anneal_strategy="cos",
            final_div_factor=100,
        )

        
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
            avg_val_loss, val_predictions = _valid_epoch(val_loader, model, criterion, model.device, writer, epoch)
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
                logger.info(f"Saved current model as best model (epoch {epoch})")

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
    config = model.config
    start = end = time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
    losses = AverageMeter()
    softmax = nn.Softmax(dim=1)
    i = 0
     # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, (X, y) in enumerate(tqdm_train_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.cuda.amp.autocast(enabled=config.AMP):
                y_preds = model(X) 
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            end = time.time()

            mse = mean_squared_error(y.detach().cpu(), softmax(y_preds).detach().cpu())
            writer.add_scalar("MSE/train", mse, epoch * total_batches + step)
            writer.add_scalar("Loss/train", loss.item(), epoch * total_batches + step)
            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(epoch+1, step, len(train_loader), 
                              remain=timeSince(start, float(step+1)/len(train_loader)),
                              loss=losses,
                              grad_norm=grad_norm,
                              lr=scheduler.get_last_lr()[0]))

    return losses.avg


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
    model.eval()
    softmax = nn.Softmax(dim=1)
    losses = AverageMeter()
    config = model.config
    prediction_dict = {}
    preds = []
    start = end = time.time()
    with tqdm(val_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
        for step, (X, y) in enumerate(tqdm_valid_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.no_grad():
                y_preds = model(X)
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy())
            end = time.time()
            mse = mean_squared_error(y.detach().cpu(), softmax(y_preds).detach().cpu())
            writer.add_scalar("MSE/val", mse,  epoch * len(val_loader) + step)
            writer.add_scalar("Loss/val", loss.item(), epoch * len(val_loader) + step)
            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(val_loader)-1):
                print('EVAL: [{0}/{1}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      .format(step, len(val_loader),
                              remain=timeSince(start, float(step+1)/len(val_loader)),
                              loss=losses))
                
    prediction_dict["predictions"] = np.concatenate(preds)
    return losses.avg, prediction_dict


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
    logger.info(f"Training Loss: {avg_train_loss:.4f}")
    logger.info(f"Validation Loss: {avg_val_loss:.4f}")


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
