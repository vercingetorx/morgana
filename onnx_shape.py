import onnxruntime as ort

path = "models/buffalo_l/det_10g.onnx"
s = ort.InferenceSession(path)
print("inputs:")
for i in s.get_inputs():
    print(i.name, i.shape)
print("outputs:")
for o in s.get_outputs():
    print(o.name, o.shape)
