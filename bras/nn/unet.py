import pytorch_lightning as pl
from monai.networks.nets import DynUnet


class DynUnet3D(DynUnet):
    """
    Create 3D model from given config parameters

    in order for U-Net to work without mismatch errors
    BraTs samples with size (240, 240, 155) should get patch
    into (128, 128, 128) before getting feed.

    config:
        - in_channels
        - out_channels
        - filters (feature maps)
        - strides
        - dropout: [True | False]
    """

    def __init__(self, config):
        super().__init__(
            spatial_dims=3,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            filters=config["filters"],
            kernel_size=config["kernels"],
            strides=config["strides"],
            upsample_kernel_size=config["strides"][1:],
            dropout=config["dropout"]
        )