import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import gc
import datetime as dtime
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, GroupKFold

import torch
from torch.utils.data import Dataset, DataLoader, Subset

# for TPU
if DEVICE == "TPU":
    import torch_xla.core.xla_model as xm

# local
from rsna.config import DEVICE, TPU
from rsna.utility import data_to_device
from rsna.dataset import RSNADataset, RSNADatasetPNG, Transform
from rsna.metrics import rsna_accuracy, rsna_roc, pfbeta


def train(model, 
          optimizer, 
          scheduler, 
          criterion, 
          df_data : pd.DataFrame, 
          cfg, 
          mlflow_client, 
          run_id : int ):
    print(">>>>> train start.")
    
    # Split in folds
    group_fold = GroupKFold(n_splits = cfg.fold)

    # Generate indices to split data into training and test set.
    k_folds = group_fold.split(X = np.zeros(len(df_data)), 
                               y = df_data['cancer'], 
                               groups = df_data['patient_id'].tolist())
    # 前処理関数の定義 (for Dataset)
    train_transform = Transform(
        horizontal_flip=cfg.horizontal_flip, 
        vertical_flip=cfg.vertical_flip,
        is_train=True
    ) 
    valid_transform = Transform(
        horizontal_flip=cfg.horizontal_flip, 
        vertical_flip=cfg.vertical_flip,
        is_train=False
    ) 

    # For each fold
    for idx_fold, (train_index, valid_index) in enumerate(k_folds):
        print(f"-------- Fold #{idx_fold}")

        # --- Read in Data ---
        train_data = df_data.iloc[train_index].reset_index(drop=True)
        valid_data = df_data.iloc[valid_index].reset_index(drop=True)

        # Create Data instances
        # train_dataset = RSNADataset(train_data, cfg.vertical_flip, cfg.horizontal_flip, cfg.csv_columns, is_train=True)
        # valid_dataset = RSNADataset(valid_data, cfg.vertical_flip, cfg.horizontal_flip, cfg.csv_columns, is_train=True)
        train_dataset = RSNADatasetPNG(train_data, train_transform, cfg.csv_columns, is_train=True)
        valid_dataset = RSNADatasetPNG(valid_data, valid_transform, cfg.csv_columns, is_train=True)
        
        # Dataloaders
        if TPU:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False) 
            train_loader = DataLoader( train_dataset, batch_size=cfg.batch_size_1, num_workers=cfg.num_workers, pin_memory=True, sampler=train_sampler)
            valid_loader = DataLoader( valid_dataset, batch_size=cfg.batch_size_2, num_workers=cfg.num_workers, pin_memory=True, sampler=valid_sampler)
        else :
            train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size_1, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
            valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size_2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True) #### shuffle = False ####

        # === EPOCHS ===
        for epoch in range(cfg.epochs):
            # print(f"Epoch # {epoch}")

            # ------------------------------------------------------
            #   TRAIN
            # ------------------------------------------------------
            # for monitorin
            list_train_preds, list_train_targets = [], []
            running_train_loss = 0.0

            # Sets the module in training mode.
            model.train()

            # For each batch                
            # for k, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            for idx, data in enumerate(train_loader):
                # Save them to device
                #   - image   : [BS, C, W, H]
                #   - meta    : [BS, 4]
                #   - targets : [BS,] <-- add a dim with unsqueeze(1) if needed
                image, meta, targets = data_to_device(data)

                # 1. clear gradients
                optimizer.zero_grad()
                # 2. forward
                #   - out : [BS, 1]
                out = model(image, meta)
                # 3. calc loss
                loss = criterion(out, targets.unsqueeze(1).float())
                # 4. Backward
                loss.backward()
                # 5. update weights
                if TPU : 
                    xm.optimizer_step(optimizer,barrier=True)
                else : 
                    optimizer.step()

                # Save information
                #   - loss
                running_train_loss += loss.item()
                #   - acc
                list_train_targets.extend(targets.cpu().numpy())
                list_train_preds.extend(torch.sigmoid(out).squeeze(1).cpu().detach().numpy()) 

                # clean memory
                del data, image, meta, targets, out, loss
                gc.collect()
                torch.cuda.empty_cache()
                # mem = psutil.virtual_memory()
                # print(f"mem = {mem.used}, {mem.available}, GPU allocated memory = {torch.cuda.memory_allocated(device=DEVICE)}")

                if cfg.debug : 
                    break # for debug

            # Compute Train Accuracy
            train_acc = rsna_accuracy(list_train_targets, list_train_preds)
            train_loss = running_train_loss / len(train_loader)
            train_pfbeta = pfbeta(list_train_targets, list_train_preds, 1)

            # mlflow logs
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_acc", train_acc, step=epoch)
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_loss", train_loss, step=epoch)
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_pfbeta", train_pfbeta, step=epoch)



            # ------------------------------------------------------
            #   EVAL
            # ------------------------------------------------------
            # for monitoring
            list_valid_targets, list_valid_preds = [], []
            running_valid_loss = 0.0

            # Sets the model in evaluation mode.
            # Disables gradients (we need to be sure no optimization happens)
            model.eval()
            with torch.no_grad():
                for idx, data in enumerate(valid_loader):
                    # Save them to device
                    image, meta, targets = data_to_device(data)
                    
                    # infer
                    out = model(image, meta)

                    # calc loss
                    loss = criterion(out, targets.unsqueeze(1).float())

                    # Save information
                    #   - loss
                    running_valid_loss += loss.item()
                    #   - acc
                    list_valid_targets.extend(targets.cpu().numpy())
                    list_valid_preds.extend(torch.sigmoid(out).squeeze(1).cpu().detach().numpy())
                   
                    # clean memory
                    del data, image, meta, targets, loss
                    gc.collect()

                    if cfg.debug : 
                        break

                # Calculate metrics (acc, roc)
                valid_loss = running_valid_loss/len(valid_loader)
                valid_acc = rsna_accuracy(list_valid_targets, list_valid_preds)
                valid_pfbeta = pfbeta(list_valid_targets, list_valid_preds, 1)

                # print
                logs_per_epoch = f'# Epoch : {epoch}/{cfg.epochs} | train loss : {train_loss :.4f}, train acc {train_acc :.4f}, train_pfbeta {train_pfbeta:.4f} | valid loss {valid_loss :.4f}, valid acc {valid_acc :.4f}, valid_pfbeta {valid_pfbeta:.4f}'
                print(logs_per_epoch)

                # mlflow logs
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_acc", valid_acc, step=epoch)
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_loss", valid_loss, step=epoch)
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_pfbeta", valid_pfbeta, step=epoch)

                # Update scheduler (for learning_rate)
                scheduler.step(valid_loss)

            # model save
            if epoch % 10 == 0 :
                model_name = f"model_fold{idx_fold+1}_epoch{epoch+1}_vacc{valid_acc:.3f}_vpfbeta{valid_pfbeta:.3f}.pth"
                if TPU:
                    xm.save(model.state_dict(), model_name)
                else:
                    torch.save(model.state_dict(), model_name)

        del train_dataset, valid_dataset, train_loader, valid_loader 
        gc.collect()