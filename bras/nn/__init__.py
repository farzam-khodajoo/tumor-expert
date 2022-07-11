import torch
from bras.nn.unet import DynUnet3D
from bras.nn.losses import DiceLoss, DiceFocalLoss

class BrainTumorInferenceMode(torch.nn.Module):
    """
    Load Unet weights and setup inference setup,
    slice 3D images using sliding windows interface from Monai.
    """

    def __init__(self) -> None:
        super().__init__()