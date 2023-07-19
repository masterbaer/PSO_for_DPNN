import numpy as np
import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    losses = torch.load("simple_gd_loss.pt")
    accuracies = torch.load("simple_gd_accuracy.pt")
    X = np.arange(0, 5000, 20)

    plt.plot(X, accuracies)
    plt.ylabel("accuracies")
    plt.savefig("simple_gd_accuracy.png")
    #plt.show()
    plt.cla()

    plt.plot(X, losses)
    plt.ylabel("losses")
    plt.savefig("simple_gd_loss.png")
    #plt.show()
    plt.cla()
