import torch
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    accuracies_true = torch.load("experiment19_accuracy_True.pt")
    accuracies_false = torch.load("experiment19_accuracy_False.pt")
    X = np.arange(0, 5000, 5)
    plt.plot(X, accuracies_true, color="r", label="using momentum on pull")
    plt.plot(X, accuracies_false, color="g", label="momentum on only gradients")
    plt.legend()
    plt.savefig("experiment19_accuracies.png")
    plt.cla()
