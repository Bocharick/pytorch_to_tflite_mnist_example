import numpy as np
import tensorflow as tf

np.set_printoptions(linewidth=np.inf)

data = np.load("output/data_x.npy")
target = np.load("output/data_y.npy")

########## Non quantized TFLite model check
# Load the TFLite model and allocate tensors.
interpreter_not_quantized = tf.lite.Interpreter(model_path='output/staticmnistnet_not_quantized.tflite')
interpreter_not_quantized.allocate_tensors()

# Get input and output tensors.
input_details = interpreter_not_quantized.get_input_details()
output_details = interpreter_not_quantized.get_output_details()
# print("input_details:", input_details, "\n")
# print("output_details:", output_details, "\n")

input_shape = input_details[0]['shape']
# print("input_shape:", input_shape)

interpreter_not_quantized.set_tensor(input_details[0]['index'], data)
interpreter_not_quantized.invoke()

non_quantized_interpreter_output_data = interpreter_not_quantized.get_tensor(output_details[0]['index'])

print("Non quantized TFLite model prediction:", "\n\tValues: " + str(non_quantized_interpreter_output_data),
      "\n\tArgmax: " + str(np.argmax(non_quantized_interpreter_output_data)))
print("\tTarget:", target.flatten())
print()

########## Quantized TFLite model check
# Load the TFLite model and allocate tensors.
interpreter_quantized = tf.lite.Interpreter(model_path='output/staticmnistnet_quantized.tflite')
interpreter_quantized.allocate_tensors()

# Get input and output tensors.
input_details = interpreter_quantized.get_input_details()
output_details = interpreter_quantized.get_output_details()
# print("input_details:", input_details, "\n")
# print("output_details:", output_details, "\n")

input_shape = input_details[0]['shape']
# print("input_shape:", input_shape)

interpreter_quantized.set_tensor(input_details[0]['index'], data)
interpreter_quantized.invoke()

non_quantized_interpreter_output_data = interpreter_quantized.get_tensor(output_details[0]['index'])

print("Quantized TFLite model prediction:", "\n\tValues: " + str(non_quantized_interpreter_output_data),
      "\n\tArgmax: " + str(np.argmax(non_quantized_interpreter_output_data)))
print("\tTarget:", target.flatten())
print()
