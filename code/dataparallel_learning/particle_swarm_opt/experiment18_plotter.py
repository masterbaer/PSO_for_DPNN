import torch
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    accuracies_true = torch.load("experiment18_accuracy_True.pt")
    accuracies_false = torch.load("experiment18_accuracy_False.pt")
    accuracies_point_five = torch.load("experiment15_accuracy_0.5.pt")

    X = np.arange(0, 5000, 20)
    plt.plot(X, accuracies_true, color="r", label="using momentum on pull")
    plt.plot(X, accuracies_false, color="g", label="momentum on only gradients")
    plt.plot(X, accuracies_point_five, color="b", label="no momentum")
    plt.legend()
    plt.savefig("experiment18_accuracies.png")
    plt.cla()
