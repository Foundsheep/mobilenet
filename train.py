import torch
from torch import nn, optim
from torchvision.datasets import ImageNet, MNIST
from torchvision.transforms import v2, ToTensor
from torch.utils.data import DataLoader
from layers import MobileNet

from datetime import datetime
from time import time
from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams


class Trainer:
    def __init__(self, width_multiplier, resolution_multiplier):
        self.mobilenet = MobileNet(width_multiplier, resolution_multiplier)
        self.cnn = MobileNet(width_multiplier, resolution_multiplier, False)

    def train(self):
        pass

    def save_model(self):
        pass

    def plot_history(self):
        pass


class DataHandler:
    def __init__(self, root):
        self.root = root
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
        ])

    def load_imagenet(self):
        ds_train = ImageNet(root=self.root, split="train")
        ds_val = ImageNet(root=self.root, split="val")
        return ds_train, ds_val


ROOT = ""
learning_rate = 1e-3
batch_size = 32
epochs = 10
width_multiplier = 1
resolution_multiplier = 1

mobilenet = MobileNet(width_multiplier, resolution_multiplier)
mobilenet_cnn = MobileNet(width_multiplier, resolution_multiplier, False)
# data_train = ImageNet(root=ROOT, split="train", transform=ToTensor())
# data_val = ImageNet(root=ROOT, split="val", transform=ToTensor())

transforms = v2.Compose([
    ToTensor(),
    v2.Lambda(lambda x: torch.cat([x] * 3, 0))
])
data_train = MNIST(root="./", download=True, train=True, transform=transforms)
data_val = MNIST(root="./", download=True, train=False, transform=transforms)
dataloader_train = DataLoader(data_train, batch_size=batch_size)
dataloader_val = DataLoader(data_val, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()  # probably needs to set something like 'logits=True'
adam_m = optim.Adam(mobilenet.parameters(), lr=learning_rate)
adam_c = optim.Adam(mobilenet_cnn.parameters(), lr=learning_rate)


def train_loop():
    history = {"loss_m": [],
               "loss_c": [],
               "time_m": [],
               "time_c": []}
    size = len(dataloader_train.dataset)
    num_batches = len(dataloader_train)
    time_m = 0
    time_c = 0

    mobilenet.train()
    mobilenet_cnn.train()
    for batch_idx, (X, y) in enumerate(dataloader_train):
        start_m = time()
        pred_m = mobilenet(X)
        loss_m = loss_fn(pred_m, y)
        loss_m.backward()
        adam_m.step()
        adam_m.zero_grad()
        end_m = time()
        time_m += (end_m - start_m)

        start_c = time()
        pred_c = mobilenet_cnn(X)
        loss_c = loss_fn(pred_c, y)
        loss_c.backward()
        adam_c.step()
        adam_c.zero_grad()
        end_c = time()
        time_c += (end_c - start_c)

        history["loss_m"].append(loss_m)
        history["time_m"].append(time_m)
        history["loss_c"].append(loss_c)
        history["time_c"].append(time_c)

        if batch_idx % 10:
            loss_m_item = loss_m.item()
            loss_c_item = loss_c.item()
            print(f"=========================\n"
                  f"--- loss_m : {loss_m_item:>5f}\n"
                  f"--- loss_c : {loss_c_item:>5f}\n"
                  f"--- [{(batch_idx + 1) * len(X):>5d} / {size:>5d}]"
                  f" time took : m = [{time_m:0.3f}s] , c = [{time_c:0.3f}]\n"
                  f"--- speed efficiency [{(time_c - time_m) / time_c * 100 :2f}%]\n")

            time_m, time_c = 0, 0
    return history


def test_loop():
    history = {"accuracy": []}
    mobilenet.eval()
    mobilenet_cnn.eval()
    size = len(dataloader_val.dataset)
    num_batches = len(dataloader_val)
    test_loss_m = 0
    test_loss_c = 0
    correct_m = 0
    correct_c = 0
    with torch.no_grad():
        for X, y in dataloader_val:
            pred_m = mobilenet(X)
            test_loss_m += loss_fn(pred_m, y).item()
            correct_m += (pred_m.argmax(1) == y).type(torch.float).sum().item()

            pred_c = mobilenet_cnn(X)
            test_loss_c += loss_fn(pred_c, y).item()
            correct_c += (pred_c.argmax(1) == y).type(torch.float).sum().item()

    print(f"=========================="
          f"--- test_loss_m : {test_loss_m/size:>5f}"
          f"--- test_loss_c : {test_loss_c/size:>5f}"
          f"--- correct_m : {correct_m:>5f}"
          f"--- correct_c : {correct_c:>5f}"
          f"--- accuracy_m : {correct_m / num_batches}"
          f"--- accuracy_c : {correct_c / num_batches}")

    history["accuracy_m"].append(correct_m / num_batches)
    history["accuracy_c"].append(correct_c / num_batches)
    return history


def run():
    total_history = {"loss_m": [],
                     "loss_c": [],
                     "time_m": [],
                     "time_c": [],
                     "accuracy_m": [],
                     "accuracy_c": []}
    for epoch in epochs:
        print(f"Epoch {epoch+1}\n")
        history_train = train_loop()
        history_test = test_loop()
        total_history["loss_m"].extend(history_train["loss_m"])
        total_history["loss_c"].extend(history_train["loss_c"])
        total_history["time_m"].extend(history_train["time_m"])
        total_history["time_c"].extend(history_train["time_c"])
        total_history["accuracy_m"].extend(history_test["accuracy_m"])
        total_history["accuracy_c"].extend(history_test["accuracy_c"])
    print("========= END ==============")

    plot(total_history)


def plot(history):
    rcParams["figure.figsize"] = 15, 16
    plt.subplot(311)

    plt.plot(history["loss_m"], label="loss_mobilenet")
    plt.plot(history["loss_c"], label="loss_mobilenet_cnn")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.title("loss comparison")

    plt.subplot(312)
    plt.plot(history["accuracy_m"], label="accuracy_mobilenet")
    plt.plot(history["accuracy_c"], label="accuracy_mobilenet_cnn")
    plt.ylabel("accuracy")
    plt.xlabel("iteration")
    plt.title("accuracy comparison")

    plt.subplot(313)
    plt.plot(history["time_m"], label="time_mobilenet")
    plt.plot(history["time_c"], label="time_mobilenet_cnn")
    plt.ylabel("seconds")
    plt.xlabel("iteration")
    plt.title("time comparison")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./plot_{now}.jpg")


if __name__ == "__main__":
    run()
