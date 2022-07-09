from monai.losses import DiceLoss


def create_loss_criterion(loss_config):
    return DiceLoss(
        include_background=loss_config["include_background"],
        sigmoid=loss_config["sigmoid"],
        batch=loss_config["batch"],
    )
