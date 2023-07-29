import os
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt


class SLDataset(Dataset):
    # assuming that transform and train will always inputs
    def __init__(self, transform=None, train=None):
        # read respective csv files if train/test
        if train:
            self.data = pd.read_csv("./mnist_sl_data/sign_mnist_train.csv")
        else:
            self.data = pd.read_csv("./mnist_sl_data/sign_mnist_test.csv")
        self.img_labels = self.data.iloc[:, 0].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels[idx]
        # get row value of item [label, pixel1, pixel2, ...] reshape into 28x28
        img_data = self.data.iloc[idx][1:].to_numpy().reshape(28, 28).astype("float32")
        # convert from numpy -> Image -> greyscale
        img_data = Image.fromarray(img_data).convert("L")
        if self.transform:
            # toTensor and normalize transforms
            image = self.transform(img_data)
        return image, label


# testing
if __name__ == "__main__":
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5)]
    )
    dataset = SLDataset(transform=transforms, train=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    train_features, train_labels = next(iter(dataloader))
    plt.imshow(train_features[0].squeeze(), cmap="gray")
    plt.show()
