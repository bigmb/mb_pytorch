from torchsummary import summary
import onnx
from onnx2pytorch import ConvertModel

__all__ = ['get_model_summary','onnx2torch']


def get_model_summary(model, input_size):
    """Prints the model summary.
    Input:
        model: PyTorch model
        input_size: input size of the model
    """
    summary(model, input_size=input_size)
    
def onnx2torch(model):
    """
    Function to convert onnx model to torch model
    Input:
        model: onnx model
    Output:
        torch model
    """
    onnx_m = onnx.load(model)
    torch_m = ConvertModel(onnx_m)
    return torch_m
