INTERPRETER=python3

time $INTERPRETER 01_mnist_train.py
time $INTERPRETER 02_pytorch_to_onnx.py
time $INTERPRETER 03_convert_onnx_to_pb.py
time $INTERPRETER 04_convert_pb_to_tflite.py
time $INTERPRETER 05_check_tflite_model.py
