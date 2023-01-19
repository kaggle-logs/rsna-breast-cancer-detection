import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

# hydra
import hydra
from omegaconf import DictConfig, OmegaConf

# local
from config import TrainConfig, DEVICE
from model import ResNet50Network
from utility import load_data, preprocess
from train import train

# set_seed()

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg : DictConfig) -> None: 
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

    train(model1, optimizer, scheduler, criterion, df_data=df_train, cfg=train_cfg)

if __name__ == "__main__" : 
    print('Device available now:', DEVICE)
    main()