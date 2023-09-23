import torch
import torch.onnx
import torchvision


torch_model = torchvision.models.resnet18(pretrained=True)
torch_model.train(False)

x = torch.randn(1, 3, 224, 224)
torch_out = torch.onnx._export(torch_model, x, "resnet18.onnx",verbose=True)


import onnx
model = onnx.load("resnet18.onnx")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)

import onnx_coreml
cml = onnx_coreml.convert(model)
cml.save('resnet18.mlmodel')