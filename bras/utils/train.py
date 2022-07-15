from typing import Tuple
import importlib
from torch import optim
from torch.nn import Module
import pytorch_lightning as pl
from bras.utils.image import expand_segmentation_as_one_hot


def create_optimizer(optimizer_config, model):
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0)
    betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           betas=betas, weight_decay=weight_decay)
    return optimizer


def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)


class LightningSegmentationModel(pl.LightningModule):
    """General Lightning module workflow, used with pl.Trainer()"""
    def __init__(self, torch_model, loss_fn, optimizer, scheduler, metric: Tuple[str, Module]) -> None:
        super().__init__()

        self.model = torch_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_tag, self.metric_fn = metric

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'monitor': self.metric_tag
        }

    def training_step(self, batch, batch_idx):
        # deconstruct data batch
        input_channels, segmentations = (
            batch['image'],
            batch['label']
        )
        # convert multi-label segmentation into multi-channel on-hot (0-1)
        segmentations = expand_segmentation_as_one_hot(segmentations, indexes=[1, 2, 4])

        output = self.model(input_channels)
        loss_value = self.loss_fn(output, segmentations)
        self.log("train_loss", loss_value)
        return loss_value

    def validation_step(self, batch, batch_idx):
        # deconstruct data batch
        input_channels, segmentations = (
            batch['image'],
            batch['label']
        )
        # convert multi-label segmentation into multi-channel on-hot (0-1)
        segmentations = expand_segmentation_as_one_hot(segmentations, indexes=[1, 2, 4])

        output = self.model(input_channels)
        metric_loss = self.metric_fn(output, segmentations)
        self.log(self.metric_tag, metric_loss)