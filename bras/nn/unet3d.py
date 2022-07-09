from xml.parsers.expat import model
from monai.networks.nets import DynUNet

def create_model(model_config):
    return DynUNet(
        spatial_dims=3,
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        kernel_size=model_config["kernels"],
        strides=model_config["strides"],
        upsample_kernel_size=model_config["strides"][1:],
        filters=model_config["filters"],
        deep_supervision=model_config["deep_supervision"]
    )