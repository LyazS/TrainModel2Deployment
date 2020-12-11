# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
import torch
dummy_input = torch.randn(1, 3, 256, 256, device='cpu')
input_names = ["input_0"]
output_names = ["output_0"]
model = None  # here is your model
torch.onnx.export(
    model,
    dummy_input,
    "net.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
)
