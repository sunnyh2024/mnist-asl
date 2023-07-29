import torch.nn as nn
import torch.nn.functional as F
import torch

# Models used for Training


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # conv -> conv -> pool -> fc -> output(fc)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # 64 * 12 * 12
        self.fc1 = nn.Linear(9216, 128)
        self.output = nn.Linear(128, 26)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # keep batch size, flatten every dim after
        x = F.relu(self.fc1(x))
        x = self.output(x)

        return x


class CNNDropout(nn.Module):
    def __init__(self, dropout_prob):
        super().__init__()
        # conv -> conv -> pool -> fc -> output(fc)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # 64 * 12 * 12
        self.fc1 = nn.Linear(9216, 128)
        self.output = nn.Linear(128, 26)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # keep batch size, flatten every dim after
        x = F.relu(F.dropout(self.fc1(x), p=self.dropout_prob))
        x = self.output(x)

        return x


class FeedForwardNN(nn.Module):
    """
    layers is an integer representing number of hidden layers in network
    layer_nodes is a list of n integers, where the ith element is the number of nodes in layer i
    """

    def __init__(self, layers, node_count):
        super(FeedForwardNN, self).__init__()
        self.act_func = nn.ReLU()
        # create list of layers (input + hidden layers + out)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(784, node_count[0]))
        for i in range(layers - 1):
            self.layers.append(nn.Linear(node_count[i], node_count[i + 1]))
        self.layers.append(nn.Linear(node_count[-1], 26))

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = x
        for layer in self.layers:
            out = self.act_func(layer(out))
        return out
