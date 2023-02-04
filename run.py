import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import mlflow
import subprocess

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf

# local
from rsna.config import TrainConfig, DEVICE, PLATFORM
from rsna.model import ResNet50Network, EfficientNet
from rsna.utility import load_data, preprocess, fix_seed
from rsna.train import train

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
    
    for key, value in cfg.items():
        client.log_param(run.info.run_id, key, value)

    yaml = OmegaConf.to_yaml(cfg)
    print(yaml)

    # model = ResNet50Network(output_size=1, num_columns=4).to(DEVICE)
    model = EfficientNet(pretrained=True).to(DEVICE)

    # ------------------

    # Run the cell below to train
    # Ran it locally on all data, see the results below

    if PLATFORM == "kaggle" : 
        df_train = load_data("train", custom_path=cfg.input_path)
    elif PLATFORM == "local" : 
        df_train = load_data("train", custom_path="/Users/ktakeda/workspace/kaggle/rsna-breast-cancer-detection/data/dicom2png_256")
    df_train = preprocess(df_train, is_train=True)

    # Tools
    # Optimizer/ Scheduler/ Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005, weight_decay=0.0)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.4)
    criterion = nn.BCEWithLogitsLoss()

    train(model, optimizer, scheduler, criterion, df_data=df_train, cfg=cfg, mlflow_client = client, run_id = run.info.run_id)

    client.set_terminated(run.info.run_id)

if __name__ == "__main__" : 
    print('Device available now:', DEVICE)
    fix_seed()
    main()
