
from torch import nn, optim
from torchvision.datasets import ImageNet
from torchvision.transforms import v2, ToTensor
from torch.utils.data import DataLoader
from layers import MobileNet


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
data_train = ImageNet(root=ROOT, split="train", transform=ToTensor())
data_val = ImageNet(root=ROOT, split="val", transform=ToTensor())
dataloader_train = DataLoader(data_train, batch_size=batch_size)
dataloader_val = DataLoader(data_val, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()  # probably needs to set something like 'logits=True'
adam_m = optim.Adam(mobilenet.parameters(), lr=learning_rate)
adam_c = optim.Adam(mobilenet_cnn.parameters(), lr=learning_rate)


def train_loop():
    size = len(dataloader_train.dataset)

    mobilenet.train()
    for batch_idx, (X, y) in enumerate(dataloader_train):
        pred_m = mobilenet(X)
        pred_c = mobilenet_cnn(X)
        loss_m = loss_fn(pred_m, y)
        loss_c = loss_fn(pred_c, y)

        loss_m.backward()
        adam_m.step()
        adam_m.zero_grad()

        loss_c.backward()
        adam_c.step()
        adam_c.zero_grad()

        if batch_idx % 10:
            loss_m_item = loss_m.item()
            loss_c_item = loss_c.item()
            print(f"=========================="
                  f"--- loss_m : {loss_m_item:>5f}"
                  f"--- loss_c : {loss_c_item:>5f}"
                  f"--- epochs : [{(batch_idx + 1) * len(X):>5d} / {size:>5d}]")





if __name__ == "__main__":
