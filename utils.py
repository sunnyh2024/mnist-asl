import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from dataset import SLDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
from models import CNN

# train function
def train(epochs, model, trainloader, optim, criterion, PATH):
    print("Training Model\n[<epoch>, <iteration>] loss = <average loss>")
    model.train()
    losses = []
    for e in range(epochs):
        iterations = 100
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data

            # prediction, forward pass
            labels_pred = model(images)

            # calculate loss
            loss = criterion(labels_pred, labels)

            # compute gradients, backward pass
            loss.backward()

            # update gradients
            optim.step()

            optim.zero_grad()

            running_loss += loss
            if i % iterations == iterations - 1:
                print(f"[{e+1}, {i+1}] loss = {running_loss / iterations:.3f}")
                losses.append(round(running_loss.item() / iterations, 3))
                running_loss = 0.0

        # save after each epoch
        torch.save(model.state_dict(), PATH)
    return losses


# test function
def test(model, testloader):
    model.eval()
    total = 0
    correct = 0
    for images, labels in testloader:
        labels_pred = model(images)
        # apply softmax and argmax to get index of class
        labels_pred = torch.argmax(F.softmax(labels_pred, dim=1), dim=1)
        correct += torch.sum(labels_pred == labels).item()
        total += labels.size(0)

    accuracy = correct / total
    return accuracy


def test_train_split(batch_size):
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5)]
    )
    train_data = SLDataset(transform=transforms, train=True)
    test_data = SLDataset(transform=transforms, train=False)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def k_folds_cross_validation(k, batch_size, model, PATH):
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5)]
    )
    total_dataset = ConcatDataset(
        [
            SLDataset(transform=transforms, train=True),
            SLDataset(transform=transforms, train=False),
        ]
    )

    # train, number of iterations through dataset, using k-folds cross validation
    datasets = random_split(total_dataset, [1 / k] * k)
    accuracies = []
    losses = []
    for fold in range(k):
        train_data = ConcatDataset(
            [datasets[i] for i in range(len(datasets)) if i != fold]
        )
        validation_data = datasets[fold]
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

        # initialize new model every fold
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # train and test for every fold
        loss = train(1, model, trainloader, optim, criterion, PATH)
        losses.extend(loss)
        accuracy = test(model, testloader)
        accuracies.append(accuracy)
        print(f'Fold {fold+1} Accuracy: {accuracy}')

    return np.mean(accuracies), losses
