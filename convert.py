import torch
from model import TrackNetV2  # Make sure to import your model definition

# Initialize the model
model = TrackNetV2(in_dim=9, out_dim=3).cuda()
model.eval()  # Set model to evaluation mode

# Define model input shape parameters
batch_size = 1
F = 3  # Number of frames
HEIGHT, WIDTH = 288, 512  # Frame dimensions

# Create a dummy input tensor with shape (batch_size, F*3, HEIGHT, WIDTH)
input_data = torch.randn(batch_size, F * 3, HEIGHT, WIDTH).cuda()

# Export the model to ONNX format with a higher opset version
torch.onnx.export(
    model,
    input_data,
    "tracknet.onnx",
    # verbose=True,
    export_params=True,
    opset_version=11,  # Try opset version 13 or higher
    do_constant_folding=False,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Optional for dynamic batch
)
