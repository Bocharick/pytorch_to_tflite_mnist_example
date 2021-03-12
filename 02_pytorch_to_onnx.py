from clean_quant_lib import *
import torch
import os
import numpy as np

np.set_printoptions(linewidth=np.inf)

TEST_BATCH_SIZE = 1
GENERATE_NEW_BATCH = False

if not os.path.isfile("output/data_x.npy") or GENERATE_NEW_BATCH:
    test_loader = get_mnist_dataset_loader("test", TEST_BATCH_SIZE)
    data, target = next(iter(test_loader))
    # print(data.shape)
    # print(target)

    np_data = data.cpu().detach().numpy()
    np_target = target.cpu().detach().numpy()
    np.save("output/data_x.npy", np_data)
    np.save("output/data_y.npy", np_target)
    print("NEW BATCH TARGET:\n", target)

data = np.load("output/data_x.npy")
target = np.load("output/data_y.npy")

data = torch.tensor(data)
target = torch.tensor(target)

pytorch_model = torch.load("output/staticmnistnet.pt", map_location='cpu')
pytorch_model.eval()
# print(pytorch_model)

prediction = pytorch_model(data)
np_prediction = prediction.cpu().detach().numpy()
print("Pytorch model prediction:", "\n\tValues: " + str(np_prediction), "\n\tArgmax: " + str(np.argmax(np_prediction)))
print("\tTarget:", target.cpu().detach().numpy().flatten())
print()

torch.onnx.export(pytorch_model,
                  (data),
                  "output/staticmnistnet.onnx",
                  verbose=False,
                  # verbose=True,
                  input_names=['input_x'],
                  output_names=['output_y']
                  )
