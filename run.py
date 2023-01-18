import subprocess
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

# local
from config import TrainConfig, DEVICE
from model import ResNet50Network
from utility import load_data, preprocess
from train import train

# set_seed()
print('Device available now:', DEVICE)

train_cfg = TrainConfig(
    fold=2,
    epochs=1,
    patience=3,
    num_workers=0,
    lr=0.005,
    wd=0.0,
    lr_patience=1,
    lr_factor=0.4,
    batch_size_1=32,
    batch_size_2=16,
    vertical_flip=0.5,
    horizontal_flip=0.5,
    output_size=1,
    csv_columns=['laterality', 'view', 'age', 'implant'],
)

VERSION = 'v1'
MODEL = 'resnet50'

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
