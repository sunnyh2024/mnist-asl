import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torch.utils.data import DataLoader, random_split, ConcatDataset
import os
import numpy as np
from utils import k_folds_cross_validation
from models import CNN, CNNDropout, FeedForwardNN
import matplotlib.pyplot as plt

# refactor to train models as needed
def main():
    models = [CNN(), CNNDropout(0.3), FeedForwardNN(2, [128, 128])]
    PATHS = [
        "./models_k_folds/cnn.pth",
        "./models_k_folds/cnn_dropout.pth",
        "./models_k_folds/ffnn.pth",
    ]
    legend = ["CNN", "CNN dropout", "FFNN"]

    for i, model in enumerate(models):
        mean_accuracy, losses = k_folds_cross_validation(4, 16, model, PATHS[i])
        plt.plot(losses)
        print(f"Mean Test Accuracy: {mean_accuracy}")
    plt.ylabel('Loss')
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.legend(legend)
    plt.savefig('./loss_plots/kfolds.png')
    plt.show()

if __name__ == "__main__":
    main()


# K-Folds Cross Validation

# CNN using Adam Optimizer and standard layers (2 conv, 1 pool, 1 fully-connected)
# Over 4 epochs using 4-Folds Cross-Validation
# average loss -> .004
# mean accuracy over validation sets -> 1.0
# accuracy on given test set = 99.78%

# CNN using Adam Optimizer and standard layers (2 conv, 1 pool, 1 fully-connected, 1 dropout(p=?))
# Over 4 epochs using 4-Fold Cross-Validation
# average loss over each epoch -> 0.112
# mean accuracy over validation sets -> .973
# accuracy on given test set = 97.3%

# FFNN using Adam Optimizer with 2 hidden layers, each 128 nodes tall
# Over 4 epochs using 4-Fold Cross-Validation
# average loss -> 1.330
# mean accuracy over validation sets -> 0.605
# accuracy on given test set = 61.24%
