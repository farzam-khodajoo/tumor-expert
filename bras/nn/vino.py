"""
https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/brain-tumor-segmentation-0002
"""
import numpy as np
from openvino.runtime import Core


def normalize(image, mask, full_intensities_range):
    ret = image.copy()
    image_masked = np.ma.masked_array(ret, ~(mask))
    ret = ret - np.mean(image_masked)
    ret = ret / np.var(image_masked) ** 0.5
    if not full_intensities_range:
        ret[ret > 5.] = 5.
        ret[ret < -5.] = -5.
        ret += 5.
        ret /= 10
        ret[~mask] = 0.
    return ret


class VinoBraTs:
    """inference model for openvin's brain tumor segmentation model"""

    def __init__(self, path_to_onnx) -> None:
        self.core = Core()
        self.model = self.core.read_model(path_to_onnx)
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.inference_requester = self.compiled_model.create_infer_request()
        self.input_tensor_idx = self.model.inputs[0].get_any_name()
        self.output_tensor = self.compiled_model.outputs[0]

    def inference(self, inputs):
        model_inputs = {self.input_tensor_idx: inputs}
        return self.inference_requester.infer(model_inputs)[self.output_tensor]
