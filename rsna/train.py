import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import gc
import datetime as dtime
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pickle

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# local
from rsna.utility import data_to_device
from rsna.dataset import RSNADataset, RSNADatasetPNG
from rsna.preprocess import Transform
from rsna.model import EfficientNet
from rsna.config import DEVICE, TPU
from rsna.metrics import rsna_accuracy, rsna_roc, pfbeta, rsna_precision_recall_f1

# for TPU
if TPU:
    import torch_xla.core.xla_model as xm


def train(df_data : pd.DataFrame, 
          cfg, 
          mlflow_client, 
          run_id : int ):
    print(">>>>> train start.")
    
    # 前処理関数の定義 (for Dataset)
    transform = Transform(cfg) 
    
    # Split in folds
    #   - GroupKFold : random_state がない、shuffle がないので、KFold を使って grouped k-fold を再現している
    kfold = KFold(n_splits=cfg.fold, shuffle=True, random_state=1993)
    unique_patient_id = df_data["patient_id"].unique()
    patient_id = df_data["patient_id"]

    # For each fold
    for idx_fold, (train_index, valid_index) in enumerate(kfold.split(unique_patient_id)):

        # --- Read in Data ---
        # index --> patient id
        train_patient_id, valid_patient_id = unique_patient_id[train_index], unique_patient_id[valid_index]
        is_train = patient_id.isin(train_patient_id)
        is_valid = patient_id.isin(valid_patient_id)

        train_data = df_data[is_train].reset_index(drop=True)
        valid_data = df_data[is_valid].reset_index(drop=True)

        with open(f"train_index_fold{idx_fold}.pkl", "wb") as f:
            pickle.dump(train_index, f)
        with open(f"valid_index_fold{idx_fold}.pkl", "wb") as f:
            pickle.dump(valid_index, f)

        print(f"-------- Fold #{idx_fold} #train={len(train_data)}, #valid={len(valid_data)}")

        # --- model init
        model = EfficientNet(model_name=cfg.model_name, pretrained=True).to(DEVICE)

        # --- Optimizer
        if cfg.optimizer.name == "Adam" :
            optimizer = torch.optim.Adam(model.parameters(), lr = cfg.optimizer.learning_rate, weight_decay = cfg.optimizer.weight_decay)
        else :
            raise NotImplementedError(cfg.optimizer.name)

        # --- Scheduler
        if cfg.scheduler.name == "ReduceLROnPlateau" :
            scheduler = ReduceLROnPlateau(optimizer=optimizer, mode=cfg.scheduler.mode, patience=cfg.scheduler.patience, verbose=True, factor=cfg.scheduler.factor)
        else :
            raise NotImplementedError(cfg.scheduler.name)

        # --- Loss
        criterion = nn.BCEWithLogitsLoss()
        
        # --- Scaler
        scaler = GradScaler()

        # Create Data instances
        train_dataset = RSNADatasetPNG(train_data, transform.get(is_train=True), cfg.csv_columns, has_target=True, image_prep_ver=cfg.preprocess.img_version)
        valid_dataset = RSNADatasetPNG(valid_data, transform.get(is_train=False), cfg.csv_columns, has_target=True, image_prep_ver=cfg.preprocess.img_version)
        
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
            list_train_preds, list_train_targets = [],[]
            dict_train_pID = {}
            running_train_loss = 0.0

            # Sets the module in training mode.
            model.train()

            # For each batch                
            # for k, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            for idx, data in enumerate(train_loader):
                with autocast(enabled=cfg.autocast):
                    # Save them to device
                    #   - image   : [BS, C, W, H]
                    #   - meta    : [BS, 4]
                    #   - targets : [BS,] <-- add a dim with unsqueeze(1) if needed
                    image, meta, targets, prediction_ids = data_to_device(data)

                    # 1. clear gradients
                    optimizer.zero_grad()
                    # 2. forward
                    #   - out : [BS, 1]
                    out = model(image, meta)
                    # 3. calc loss
                    loss = criterion(out, targets.unsqueeze(1).float())
                
                # 4. Backward
                scaler.scale(loss).backward()
                # 5. update weights
                if TPU : 
                    xm.optimizer_step(optimizer,barrier=True)
                else : 
                    scaler.step(optimizer)

                scaler.update()

                # Save information
                #   - loss
                running_train_loss += loss.item()
                #   - acc
                list_train_targets.extend(targets.cpu().numpy())
                list_train_preds.extend(torch.sigmoid(out).squeeze(1).cpu().detach().numpy()) 
                #   - prediction_id
                targets = targets.cpu().numpy()
                preds = torch.sigmoid(out).squeeze(1).cpu().detach().numpy()
                for prediction_id, target, pred in zip(prediction_ids, targets, preds) : 
                    if not dict_train_pID.get(prediction_id, False) :
                        # 同じ prediciton_id に対して y_true は1種類であると決め打ちしている
                        # （同じprediction_idに別のy_trueが振られていることがある？？）
                        dict_train_pID[prediction_id] = {"target" : target, "pred" : [pred,]}
                    else :
                        dict_train_pID[prediction_id]["pred"].append(pred)

                # clean memory
                del data, image, meta, targets, out, loss
                gc.collect()
                torch.cuda.empty_cache()
                # mem = psutil.virtual_memory()
                # print(f"mem = {mem.used}, {mem.available}, GPU allocated memory = {torch.cuda.memory_allocated(device=DEVICE)}")

                if cfg.debug: 
                    break # for debug

            # Compute Train Accuracy
            train_acc = rsna_accuracy(list_train_targets, list_train_preds)
            train_loss = running_train_loss / len(train_loader)
            train_pfbeta = pfbeta(list_train_targets, list_train_preds, 1)
            train_precision, train_recall, train_f1 = rsna_precision_recall_f1(list_train_targets, list_train_preds)
            
            # create dict
            list_train_pID_target = []
            list_train_pID_pred_max, list_train_pID_pred_min, list_train_pID_pred_median, list_train_pID_pred_mean = [], [], [], []
            for k, v in dict_train_pID.items():
                list_train_pID_target.append(v["target"])
                list_train_pID_pred_max.append(np.max(v["pred"]))
                list_train_pID_pred_min.append(np.min(v["pred"]))
                list_train_pID_pred_median.append(np.median(v["pred"]))
                list_train_pID_pred_mean.append(np.mean(v["pred"]))

            # mlflow logs
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_acc", train_acc, step=epoch)
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_loss", train_loss, step=epoch)
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_pfbeta", train_pfbeta, step=epoch)
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_precision", train_precision, step=epoch)
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_recall", train_recall, step=epoch)
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_f1", train_f1, step=epoch)
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_prediction_id_pfbeta_max", pfbeta(list_train_pID_target, list_train_pID_pred_max, 1), step=epoch)
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_prediction_id_pfbeta_min", pfbeta(list_train_pID_target, list_train_pID_pred_min, 1), step=epoch)
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_prediction_id_pfbeta_medium",pfbeta(list_train_pID_target, list_train_pID_pred_median, 1),  step=epoch)
            mlflow_client.log_metric(run_id, f"{idx_fold}fold_train_prediction_id_pfbeta_mean",pfbeta(list_train_pID_target, list_train_pID_pred_mean, 1),  step=epoch)



            # ------------------------------------------------------
            #   EVAL
            # ------------------------------------------------------
            # for monitoring
            list_valid_targets, list_valid_preds = [], []
            dict_valid_pID = {}
            running_valid_loss = 0.0

            # Sets the model in evaluation mode.
            # Disables gradients (we need to be sure no optimization happens)
            model.eval()
            with torch.no_grad():
                for idx, data in enumerate(valid_loader):

                    with autocast(enabled=cfg.autocast):
                        # Save them to device
                        image, meta, targets, prediction_ids = data_to_device(data)
                    
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
                        #   - prediction_id
                        targets = targets.cpu().numpy()
                        preds = torch.sigmoid(out).squeeze(1).cpu().detach().numpy()
                        for prediction_id, target, pred in zip(prediction_ids, targets, preds) : 
                            if not dict_valid_pID.get(prediction_id, False) :
                                # 同じ prediciton_id に対して y_true は1種類であると決め打ちしている
                                # （同じprediction_idに別のy_trueが振られていることがある？？）
                                dict_valid_pID[prediction_id] = {"target" : target, "pred" : [pred,]}
                            else :
                                dict_valid_pID[prediction_id]["pred"].append(pred)
                   
                    # clean memory
                    del data, image, meta, targets, loss
                    gc.collect()

                    if cfg.debug: 
                        break

                # Calculate metrics (acc, roc)
                valid_loss = running_valid_loss/len(valid_loader)
                valid_acc = rsna_accuracy(list_valid_targets, list_valid_preds)
                valid_pfbeta = pfbeta(list_valid_targets, list_valid_preds, 1)
                valid_precision, valid_recall, valid_f1 = rsna_precision_recall_f1(list_valid_targets, list_valid_preds)
            
                # create dict
                list_valid_pID_target = []
                list_valid_pID_pred_max, list_valid_pID_pred_min, list_valid_pID_pred_median, list_valid_pID_pred_mean = [], [], [], []
                for k, v in dict_valid_pID.items():
                    list_valid_pID_target.append(v["target"])
                    list_valid_pID_pred_max.append(np.max(v["pred"]))
                    list_valid_pID_pred_min.append(np.min(v["pred"]))
                    list_valid_pID_pred_median.append(np.median(v["pred"]))
                    list_valid_pID_pred_mean.append(np.mean(v["pred"]))

                # print
                logs_per_epoch = f'# Epoch : {epoch}/{cfg.epochs} | train loss : {train_loss :.4f}, train acc {train_acc :.4f}, train_pfbeta {train_pfbeta:.4f} | valid loss {valid_loss :.4f}, valid acc {valid_acc :.4f}, valid_pfbeta {valid_pfbeta:.4f}'
                print(logs_per_epoch)
                print(f'train pfbeta = {pfbeta(list_train_pID_target, list_train_pID_pred_max, 1)}, valid pfbeta = {pfbeta(list_valid_pID_target, list_valid_pID_pred_max, 1)}')

                # mlflow logs
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_acc", valid_acc, step=epoch)
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_loss", valid_loss, step=epoch)
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_pfbeta", valid_pfbeta, step=epoch)
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_precision", valid_precision, step=epoch)
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_recall", valid_recall, step=epoch)
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_f1", valid_f1, step=epoch)
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_prediction_id_pfbeta_max", pfbeta(list_valid_pID_target, list_valid_pID_pred_max, 1), step=epoch)
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_prediction_id_pfbeta_min", pfbeta(list_valid_pID_target, list_valid_pID_pred_min, 1), step=epoch)
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_prediction_id_pfbeta_medium",pfbeta(list_valid_pID_target, list_valid_pID_pred_median, 1),  step=epoch)
                mlflow_client.log_metric(run_id, f"{idx_fold}fold_valid_prediction_id_pfbeta_mean",pfbeta(list_valid_pID_target, list_valid_pID_pred_mean, 1),  step=epoch)

                # Update scheduler (for learning_rate)
                scheduler.step(valid_loss)

            # metrics save
            with open(f"metrics/f{idx_fold+1}_dict_train_pID.pkl", "wb") as f:
                pickle.dump(dict_train_pID, f)
            with open(f"metrics/f{idx_fold+1}_dict_valid_pID.pkl", "wb") as f:
                pickle.dump(dict_valid_pID, f)

            # model save
            model_name = f"models/model_fold{idx_fold+1}_epoch{epoch+1}_vacc{valid_acc:.3f}_vpfbeta{valid_pfbeta:.3f}.pth"
            if TPU:
                xm.save(model.state_dict(), model_name)
            else:
                torch.save(model.state_dict(), model_name)

        del train_dataset, valid_dataset, train_loader, valid_loader, model
        gc.collect()

        # 途中で fold を break する
        #   - k-fold の設定のまま、擬似的に hold_out を再現する (ex. -1 なら test_size = 1/k の hold_out)
        if idx_fold > cfg.fold_break : 
            break
