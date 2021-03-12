import onnx
from onnx_tf.backend import prepare
import numpy as np

np.set_printoptions(linewidth=np.inf)

onnx_model = onnx.load("output/staticmnistnet.onnx")  # load onnx model
# print(onnx_model)

data = np.load("output/data_x.npy")
target = np.load("output/data_y.npy")

prepared_model = prepare(onnx_model)
# print(prepared_model)

output = prepared_model.run(data)  # run the loaded model
np_prediction = output["output_y"]

print("ONNX model prediction:", "\n\tValues: " + str(np_prediction), "\n\tArgmax: " + str(np.argmax(np_prediction)))
print("\tTarget:", target.flatten())
print()

prepared_model.export_graph("output/onnx_exported_to_pb")
