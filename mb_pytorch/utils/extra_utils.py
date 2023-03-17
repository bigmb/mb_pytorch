from torchsummary import summary

__all__ = ['get_model_summary']


def get_model_summary(model, input_size):
    """Prints the model summary.
    Input:
        model: PyTorch model
        input_size: input size of the model
    """
    summary(model, input_size=input_size)
    