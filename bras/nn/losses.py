from monai.losses import DiceLoss, DiceFocalLoss


class BraTsDiceLoss(DiceLoss):
    """
    Compute average Dice loss between two tensors.
    It can support both multi-classes and multi-labels tasks.
    The data input (BNHW[D] where N is number of classes) is compared with ground truth target (BNHW[D]).
    """

    def __init__(self, config):
        super().__init__(
            include_background=config["background"],
            sigmoid=config["sigmoid"],
            smooth_dr=config["smooth_dr"],
            smooth_nr=0,
            batch=config["batch"]
        )


class BraTsDiceFocalLoss(DiceFocalLoss):
    """
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in monai.losses.DiceLoss.
    The details of Focal Loss is shown in monai.losses.FocalLoss.
    """

    def __init__(self, config):
        super().__init__(
            include_background=config["background"],
            sigmoid=config["sigmoid"],
            smooth_dr=config["smooth_dr"],
            smooth_nr=0,
            batch=config["batch"], 
            gamma=config["gamma"]
        )