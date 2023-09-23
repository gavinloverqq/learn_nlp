import torch
import torch.onnx
import torchvision


torch_model = torchvision.models.alexnet(pretrained=True)
torch_model.train(False)

x = torch.randn(1, 3, 224, 224)
torch_out = torch.onnx._export(torch_model, x, "alexnet.onnx",verbose=True)


import onnx
model = onnx.load("alexnet.onnx")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)


import numpy as np
import caffe2.python.onnx.backend as onnx_caffe2_backend

prepared_backend = onnx_caffe2_backend.prepare(model)
W = {model.graph.input[0].name: x.data.numpy()}
c2_out = prepared_backend.run(W)[0]

np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)
print("Exported model has been executed on Caffe2 backend, and the result looks good!")