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

# local
from rsna.config import TrainConfig, DEVICE
from rsna.utility import data_to_device
from rsna.dataset import RSNADataset, RSNADatasetPNG
from rsna.metrics import rsna_accuracy, rsna_roc


def train(model, 
          optimizer, 
          scheduler, 
          criterion, 
          df_data : pd.DataFrame, 
          cfg : TrainConfig, 
          mlflow_client, 
          run_id : int ):
    print(">>>>> train start.")
    
    # Split in folds
    group_fold = GroupKFold(n_splits = cfg.fold)

    # Generate indices to split data into training and test set.
    k_folds = group_fold.split(X = np.zeros(len(df_data)), 
                               y = df_data['cancer'], 
                               groups = df_data['patient_id'].tolist())
    
    # For each fold
    for fold, (train_index, valid_index) in enumerate(k_folds):
        print(f"-------- Fold #{fold}")

        # --- Create Instances ---
        # Best ROC score in this fold
        best_roc = None
        # Reset patience before every fold
        patience_f = cfg.patience

        # --- Read in Data ---
        train_data = df_data.iloc[train_index].reset_index(drop=True)
        valid_data = df_data.iloc[valid_index].reset_index(drop=True)

        # Create Data instances
        # train_dataset = RSNADataset(train_data, cfg.vertical_flip, cfg.horizontal_flip, cfg.csv_columns, is_train=True)
        # valid_dataset = RSNADataset(valid_data, cfg.vertical_flip, cfg.horizontal_flip, cfg.csv_columns, is_train=True)
        train_dataset = RSNADatasetPNG(train_data, cfg.vertical_flip, cfg.horizontal_flip, cfg.csv_columns, is_train=True)
        valid_dataset = RSNADatasetPNG(valid_data, cfg.vertical_flip, cfg.horizontal_flip, cfg.csv_columns, is_train=True)
        
        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size_1, shuffle=True, num_workers=cfg.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size_2, shuffle=False, num_workers=cfg.num_workers) #### shuffle = False ####

        # === EPOCHS ===
        for epoch in range(cfg.epochs):
            print(f"Epoch # {epoch}")

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

                # if idx > 5: break # for debug

            # Compute Train Accuracy
            train_acc = rsna_accuracy(list_train_targets, list_train_preds)
            train_loss = running_train_loss / len(train_loader)

            # mlflow logs
            mlflow_client.log_metric(run_id, f"{idx}fold_train_acc", train_acc, step=epoch)
            mlflow_client.log_metric(run_id, f"{idx}fold_train_loss", train_loss, step=epoch)



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

                    # if idx > 5 : break # for debug

                # Calculate metrics (acc, roc)
                valid_loss = running_valid_loss/len(valid_loader)
                valid_acc = rsna_accuracy(list_valid_targets, list_valid_preds)

                # print
                logs_per_epoch = f'Epoch : {epoch}/{cfg.epochs} | train loss : {train_loss :.4f}, train acc {train_acc :.4f}, valid loss {valid_loss :.4f}, valid acc {valid_acc :.4f}'
                print(logs_per_epoch)

                # mlflow logs
                mlflow_client.log_metric(run_id, f"{idx}fold_valid_acc", valid_acc, step=epoch)
                mlflow_client.log_metric(run_id, f"{idx}fold_valid_loss", valid_loss, step=epoch)

                # Update scheduler (for learning_rate)
                scheduler.step(valid_loss)

                # model save
                model_name = f"model_fold{idx+1}_epoch{epoch+1}_validacc{valid_acc:.3f}.pth"
                torch.save(model.state_dict(), model_name)

        del train_dataset, valid_dataset, train_loader, valid_loader 
        gc.collect()