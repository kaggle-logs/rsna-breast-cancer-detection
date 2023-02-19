import os
import torch
import mlflow
import subprocess

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf

# local
from rsna.config import DEVICE, PLATFORM
from rsna.model import ResNet50Network, EfficientNet
from rsna.utility import load_data, fix_seed
from rsna.preprocess import df_preprocess
from rsna.train import train

# set_seed()

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg : DictConfig) -> None: 

    # --- mlflow start
    client = mlflow.tracking.MlflowClient()
    # create exp
    if PLATFORM == "kaggle" : 
        experiment_name = "kaggle"
    elif PLATFORM == "local" : 
        experiment_name = "local"

    try:
        exp_id = client.create_experiment(experiment_name)
    except:
        exp_id = client.get_experiment_by_name(experiment_name).experiment_id
    
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

    # ------------------

    # Run the cell below to train
    # Ran it locally on all data, see the results below

    if PLATFORM == "kaggle" : 
        df_train = load_data("train", custom_path=cfg.dataset.input_path, external=cfg.dataset.external)
    elif PLATFORM == "local" : 
        df_train = load_data("train", custom_path="/Users/ktakeda/workspace/kaggle/rsna-breast-cancer-detection/data/dicom2png_512", external=cfg.dataset.external)
    df_train = df_preprocess(df_train, is_train=True, sampling=cfg.preprocess.sampling)

    train(df_data=df_train, cfg=cfg, mlflow_client = client, run_id = run.info.run_id)

    client.set_terminated(run.info.run_id)

if __name__ == "__main__" : 
    print('Device available now:', DEVICE)
    fix_seed()
    # output dir
    os.makedirs("models/", exist_ok=True)
    os.makedirs("logs/", exist_ok=True)
    os.makedirs("logs/metrics", exist_ok=True)
    # 
    main()
