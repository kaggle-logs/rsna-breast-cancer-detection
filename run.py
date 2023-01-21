import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import mlflow
import subprocess

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf

# local
from config import TrainConfig, DEVICE, PLATFORM
from model import ResNet50Network
from utility import load_data, preprocess
from train import train

# set_seed()

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg : DictConfig) -> None: 

    # --- mlflow start
    client = mlflow.tracking.MlflowClient()
    # create exp
    experiment = "experiment"
    try:
        exp_id = client.create_experiment(experiment)
    except:
        exp_id = client.get_experiment_by_name(experiment).experiment_id
    
    # MLFlow system tags 
    # - https://mlflow.org/docs/latest/tracking.html?highlight=commit#system-tags
    if PLATFORM == "kaggle" : 
        res = subprocess.run("cd rsna-breast-cancer-detection && git rev-parse HEAD", shell=True, capture_output=True)
        tags = {"mlflow.source.git.commit" : res.stdout.decode("utf-8") }
    elif PLATFORM == "local":
        tags = {"mlflow.source.git.commit" : subprocess.check_output("git rev-parse HEAD".split()).strip().decode("utf-8") }
    run = client.create_run(exp_id, tags=tags)

    client.log_metric(run.info.run_id, "fold", cfg.exp.fold)

    yaml = OmegaConf.to_yaml(cfg)
    print(yaml)

    train_cfg = TrainConfig(
        fold = cfg.exp.fold,
        epochs = cfg.exp.epochs,
        patience = cfg.exp.patience,
        num_workers = cfg.exp.num_workers,
        lr = cfg.exp.lr,
        wd = cfg.exp.wd,
        lr_patience = cfg.exp.lr_patience,
        lr_factor = cfg.exp.lr_factor,
        batch_size_1 = cfg.exp.batch_size_1,
        batch_size_2 = cfg.exp.batch_size_2,
        vertical_flip = cfg.exp.vertical_flip,
        horizontal_flip = cfg.exp.horizontal_flip,
        output_size = cfg.exp.output_size,
        csv_columns = cfg.exp.csv_columns,
    )

    model1 = ResNet50Network(output_size=1,
                             num_columns=4).to(DEVICE)

    # ------------------

    # Run the cell below to train
    # Ran it locally on all data, see the results below
    df_train = load_data("train")
    df_train = preprocess(df_train, is_train=True)

    # Tools
    # Optimizer/ Scheduler/ Criterion
    optimizer = torch.optim.Adam(model1.parameters(), lr = 0.005, weight_decay=0.0)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.4)
    criterion = nn.BCEWithLogitsLoss()

    train(model1, optimizer, scheduler, criterion, df_data=df_train, cfg=train_cfg, mlflow_client = client, run_id = run.info.run_id)

    client.set_terminated(run.info.run_id)

if __name__ == "__main__" : 
    print('Device available now:', DEVICE)
    main()
