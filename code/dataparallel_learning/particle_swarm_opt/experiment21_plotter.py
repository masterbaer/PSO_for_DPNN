import torch
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    accuracies = torch.load("experiment21_accuracy.pt")
    accuracies_sequential = torch.load("simple_gd_accuracy_0.1_256.pt")  # produced in sequential learning folder

    X = np.arange(0, 5000, 20)
    plt.plot(X, accuracies, label="pso_average_pull_v2", color="r")
    plt.plot(X, accuracies_sequential, label="seq", color="g")
    plt.legend()
    plt.savefig("experiment21_accuracies.png")
    plt.cla()
