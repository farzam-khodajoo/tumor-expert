from torch.nn import Module
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete
)
from monai.metrics import DiceMetric
from monai.data import decollate_batch


class DiceLightningMetric(DiceMetric):
    """Compute average dice loss for model evaluation"""

    post_trans = Compose(
        [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )

    def __init__(self) -> None:
        super().__init__(
            include_background=True,
            reduction="mean"
        )

    def forward(self, y_pred, y):
        y_pred = [self.post_trans(i) for i in decollate_batch(y_pred)]
        self(y_pred=y_pred, y=y)
        metric = self.aggregate().item()
        self.reset()
        return metric
