

# Optimizer/ Scheduler/ Criterion
optimizer = torch.optim.Adam(model.parameters(), lr = LR, 
                             weight_decay=WD)
scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', 
                              patience=LR_PATIENCE, verbose=True, factor=LR_FACTOR)
criterion = nn.BCEWithLogitsLoss()


def train(model, optimizer, scheduler, criterion, train_data : pd.DataFrame, cfg : TrainConfig):
    
    # Split in folds
    group_fold = GroupKFold(n_splits = cfg.fold)

    # Generate indices to split data into training and test set.
    k_folds = group_fold.split(X = np.zeros(len(train_data)), 
                               y = train_data['cancer'], 
                               groups = train_data['patient_id'].tolist())
    
    # For each fold
    for idx, (train_index, valid_index) in enumerate(k_folds):

        # --- Create Instances ---
        # Best ROC score in this fold
        best_roc = None
        # Reset patience before every fold
        patience_f = PATIENCE

        # --- Read in Data ---
        train_data = train_data.iloc[train_index].reset_index(drop=True)
        valid_data = train_data.iloc[valid_index].reset_index(drop=True)

        # Create Data instances
        train_dataset = RSNADataset(train_data, cfg.vertical_flip, cfg.horizontal_flip, 
                                    is_train=True)
        valid_dataset = RSNADataset(valid_data, cfg.vertical_flip, cfg.horizontal_flip,
                                    is_train=True)
        
        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size_1, 
                                  shuffle=True, num_workers=cfg.num_worker2)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size_2, 
                                  shuffle=False, num_workers=cfg.num_workers)

        # === EPOCHS ===
        for epoch in range(cfg.epochs):
            correct = 0
            train_losses = 0

            # === TRAIN ===
            # Sets the module in training mode.
            model.train()

            # For each batch                
            for k, data in tqdm(enumerate(train_loader)):
                # Save them to device
                image, meta, targets = data_to_device(data)

                # Clear gradients first; very important
                # usually done BEFORE prediction
                optimizer.zero_grad()

                # Forward
                out = model(image, meta)

                # Calc loss
                loss = criterion(out, targets.unsqueeze(1).float())

                # Backward
                loss.backward()

                optimizer.step()

                # --- Save information after this batch ---
                # Save loss
                train_losses += loss.item()
                # From log probabilities to actual probabilities
                train_preds = torch.round(torch.sigmoid(out)) # 0 and 1
                # Number of correct predictions
                correct += (train_preds.cpu() == targets.cpu().unsqueeze(1)).sum().item()

            # Compute Train Accuracy
            train_acc = correct / len(train_index)

            # train loop fin
            # -------------------------

            # === EVAL ===
            # Sets the model in evaluation mode.
            model.eval()

            # Create matrix to store evaluation predictions (for accuracy)
            valid_preds = torch.zeros(size = (len(valid_index), 1), 
                                      device=DEVICE, dtype=torch.float32)

            # Disables gradients (we need to be sure no optimization happens)
            with torch.no_grad():
                for k, data in tqdm(enumerate(valid_loader)):
                    # Save them to device
                    image, meta, targets = data_to_device(data)
                    out = model(image, meta)
                    pred = torch.sigmoid(out)
                    valid_preds[k*image.shape[0] : k*image.shape[0] + image.shape[0]] = pred

                # Calculate accuracy
                valid_acc = accuracy_score(valid_data['cancer'].values, 
                                           torch.round(valid_preds.cpu()))
                # Calculate ROC
                valid_roc = roc_auc_score(valid_data['cancer'].values, 
                                          valid_preds.cpu())
               # Calculate time on Train + Eval
                duration = str(dtime.timedelta(seconds=time() - start_time))[:7]

                # PRINT INFO
                final_logs = '{} | Epoch: {}/{} | Loss: {:.4} | Acc_tr: {:.3} | Acc_vd: {:.3} | ROC: {:.3}'.\
                                format(duration, epoch+1, EPOCHS, 
                                       train_losses, train_acc, valid_acc, valid_roc)
                print(final_logs)

                # === SAVE MODEL ===

                # Update scheduler (for learning_rate)
                scheduler.step(valid_roc)
                # Name the model
                model_name = f"Fold{i+1}_Epoch{epoch+1}_ValidAcc{valid_acc:.3f}_ROC{valid_roc:.3f}.pth"

                # Update best_roc
                if not best_roc: # If best_roc = None
                    best_roc = valid_roc
                    torch.save(model.state_dict(), model_name)
                    continue

                if valid_roc > best_roc:
                    best_roc = valid_roc
                    # Reset patience (because we have improvement)
                    patience_f = PATIENCE
                    torch.save(model.state_dict(), model_name)
                else:
                    # Decrease patience (no improvement in ROC)
                    patience_f = patience_f - 1
                    if patience_f == 0:
                        stop_logs = 'Early stopping (no improvement since 3 models) | Best ROC: {}'.\
                                    format(best_roc)
                        add_in_file(stop_logs, f)
                        print(stop_logs)
                        break

        # === CLEANING ===
        # Clear memory
        del train, valid, train_loader, valid_loader, image, targets
        gc.collect()