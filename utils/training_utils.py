
import gc
import numpy as np
import time
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets.data_loader import CustomDataset
from generics import Generics, Paths
import torch.nn.functional as F
from utils.general_utils import AverageMeter, get_logger, timeSince

def train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device = Generics.DEVICE):
    """One epoch training pass."""
    model.train()
    config = model.config
    scaler = torch.cuda.amp.GradScaler(enabled=model.config.AMP)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0

    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, (X, y) in enumerate(tqdm_train_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.cuda.amp.autocast(enabled=config.AMP):
                X = X.to(device)
                y_preds = model(X)
                y_preds = y_preds.to(device)
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
                global_step += 1
                scheduler.step()
            end = time.time()

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


def valid_epoch(valid_loader, model, criterion):
    model.eval()
    config = model.config
    device = model.device
    softmax = nn.Softmax(dim=1)
    losses = AverageMeter()
    prediction_dict = {}
    preds = []
    start = end = time.time()
    with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
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

            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(valid_loader)-1):
                print('EVAL: [{0}/{1}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      .format(step, len(valid_loader),
                              remain=timeSince(start, float(step+1)/len(valid_loader)),
                              loss=losses))

    prediction_dict["predictions"] = np.concatenate(preds)
    return losses.avg, prediction_dict
  
  
  
  
def train_loop(train_dataset: CustomDataset, val_dataset: CustomDataset, model, fold: int):
    logger = get_logger("utils/training_utils/train_loop")
    logger.info(f"========== Fold: {fold} training ==========")
    best_loss = np.inf
    model_config = model.config
    criterion = nn.KLDivLoss(reduction=model_config.KLDIV_REDUCTION)
    if model_config.OPTIMIZER == "adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=model_config.LEARNING_RATE)
    
    if model_config.SCHEDULER == "CosineAnnealingLR":
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, model_config.T_MAX)
    target_preds = pd.read_csv(Paths.TRAIN_CSV)
    train_loader = train_dataset.get_torch_data_loader()
    val_loader = val_dataset.get_torch_data_loader()
    # ====== ITERATE EPOCHS ========
    for epoch in range(model.config.EPOCHS):
        start_time = time.time()

        # ======= TRAIN ==========
        
        avg_train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler) 

        # ======= EVALUATION ==========
        
        avg_val_loss, prediction_dict = valid_epoch(val_loader, model, criterion)
        predictions = prediction_dict["predictions"]

        # ======= SCORING ==========
        elapsed = time.time() - start_time

        logger.info(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            logger.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        Generics.Paths.BEST_MODEL_CHECKPOINTS + f"/{train_dataset.config.__class__.__name__}/{model.config.__class__.__name__}/dataset_{train_dataset.config.__class__.__name__}_model_{model.config.__class__.__name__}_fold_{fold}_best.pth")

        predictions = torch.load(Generics.Paths.BEST_MODEL_CHECKPOINTS + f"/{train_dataset.config.__class__.__name__}/{model.config.__class__.__name__}/dataset_{train_dataset.config.__class__.__name__}_model_{model.config.__class__.__name__}_fold_{fold}_best.pth",
                                map_location=torch.device('cpu'))['predictions']
        val_loader[target_preds] = predictions


    torch.cuda.empty_cache()
    gc.collect()

    return val_loader
  
  

def get_result(df, label_cols, target_preds):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(df[label_cols].values)
    preds = torch.tensor(df[target_preds].values)
    preds = F.log_softmax(preds, dim=1)
    result = kl_loss(preds, labels)
    return result