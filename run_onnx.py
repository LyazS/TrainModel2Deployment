import numpy as np
import onnxruntime as ort
onnx_path = ""
ort_sess = ort.InferenceSession(onnx_path)
x = np.random.randn(1, 3, 256, 256)
x = x.astype(np.float32)
onnx_out = ort_sess.run(None, {"input_0": x})