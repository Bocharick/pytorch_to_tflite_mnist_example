from clean_quant_lib import *
import torch
from types import SimpleNamespace
import os

output_directory = "output"
os.makedirs(output_directory, exist_ok=True)

BATCH_SIZE = 250
TEST_BATCH_SIZE = 2000
EPOCHS = 5  # 20
SEED = 158123

LEARNING_RATE = 1.0
GAMMA = 0.7

torch.manual_seed(SEED)
device = torch.device("cpu")

train_loader = get_mnist_dataset_loader("train", BATCH_SIZE)
test_loader = get_mnist_dataset_loader("test", TEST_BATCH_SIZE)

args = SimpleNamespace(batch_size=BATCH_SIZE,
                       dry_run=False,
                       epochs=EPOCHS,
                       gamma=GAMMA,
                       log_interval=10,
                       lr=LEARNING_RATE,
                       no_cuda=False,
                       save_model=True,
                       seed=SEED,
                       test_batch_size=TEST_BATCH_SIZE)

###################################################################
# StaticMnistNet train
###################################################################
staticmnistnet_model = StaticMnistNet().to(device)
optimizer = optim.Adadelta(staticmnistnet_model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

for epoch in range(1, EPOCHS + 1):
    train(args, staticmnistnet_model, device, train_loader, optimizer, epoch)
    test(staticmnistnet_model, device, test_loader)
    scheduler.step()

if args.save_model:
    # torch.save(staticmnistnet_model.state_dict(), os.path.join(output_directory, "staticmnistnet_state_dict.pt"))
    torch.save(staticmnistnet_model, os.path.join(output_directory, "staticmnistnet.pt"))
