import torch
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    accuracies = torch.load("experiment18_accuracy.pt")
    X = np.arange(0, 5000, 5)
    plt.plot(X, accuracies)
    plt.savefig("experiment18_accuracies.png")

