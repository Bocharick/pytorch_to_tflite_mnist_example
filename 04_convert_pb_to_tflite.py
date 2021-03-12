import tensorflow as tf

export_dir = "output/onnx_exported_to_pb"

# Convert the model to tflite without quantization
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)  # path to the SavedModel directory
tflite_model = converter.convert()
# Save the model.
with open('output/staticmnistnet_not_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert the model to tflite with quantization
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)  # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
# Save the model.
with open('output/staticmnistnet_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
