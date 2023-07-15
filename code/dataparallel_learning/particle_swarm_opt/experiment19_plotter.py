import torch
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    accuracies = torch.load("experiment19_accuracy_True.pt")
    X = np.arange(0, 5000, 5)
    plt.plot(X, accuracies)
    plt.savefig("experiment19_accuracy_True.png")

    losses = torch.load("experiment19_loss_True.pt")
    X = np.arange(0, 5000, 5)
    plt.plot(X, losses)
    plt.savefig("experiment19_loss_True.png")