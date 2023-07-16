import torch
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    accuracies_true = torch.load("experiment19_accuracy_True.pt")
    X = np.arange(0, 5000, 5)
    plt.plot(X, accuracies_true)
    plt.savefig("experiment19_accuracy_True.png")
    plt.cla()

    losses_true = torch.load("experiment19_loss_True.pt")
    X = np.arange(0, 5000, 5)
    plt.plot(X, losses_true)
    plt.savefig("experiment19_loss_True.png")
    plt.cla()

    accuracies_false = torch.load("experiment19_accuracy_False.pt")
    X = np.arange(0, 5000, 5)
    plt.plot(X, accuracies_false)
    plt.savefig("experiment19_accuracy_False.png")
    plt.cla()

    losses_false = torch.load("experiment19_loss_False.pt")
    X = np.arange(0, 5000, 5)
    plt.plot(X, losses_false)
    plt.savefig("experiment19_loss_False.png")
    plt.cla()
