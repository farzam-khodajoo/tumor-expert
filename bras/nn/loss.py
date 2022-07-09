from math import gamma
import torch.nn as nn
from monai.losses import DiceLoss


def create_loss_citeration(loss_config):
    return DiceLoss(
        include_background=loss_config["include_background"],
        sigmoid=loss_config["sigmoid"],
        batch=loss_config["batch"],
        gamma=loss_config["gamma"]
    )