import torch
import torch.nn as nn

from rsna.config import DEVICE

def get_loss(cfg):
    if cfg.loss.name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.loss.pos_weight).to(DEVICE))
    else :
        raise NotImplemented()