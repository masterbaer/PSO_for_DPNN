import torch
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    losses_zero = torch.load("experiment15_loss_0.0.pt")  # (particle,iteration)
    accuracies_zero = torch.load("experiment15_accuracy_0.0.pt")

    losses_point_five = torch.load("experiment15_loss_0.5.pt")
    accuracies_point_five = torch.load("experiment15_accuracy_0.5.pt")

    losses_one = torch.load("experiment15_loss_1.0.pt")
    accuracies_one = torch.load("experiment15_accuracy_1.0.pt")

    accuracies_sequential = torch.load("simple_gd_accuracy.pt") # produced in sequential learning folder

    accuracies_sequential = accuracies_sequential[:1000]

    X = np.arange(0, 5000, 5)

    plt.plot(X, accuracies_zero, color="r", label="0", linewidth=1)
    plt.plot(X, accuracies_point_five, color="g", label="0.5", linewidth=1)
    plt.plot(X, accuracies_one, color="b", label="1", linewidth=1)
    plt.plot(X, accuracies_sequential, color="y", label="seq", linewidth = 1)
    plt.ylabel("accuracies")
    plt.xlabel("iterations")
    plt.legend()
    plt.savefig("experiment15_accuracy.png")
    plt.cla()

    plt.plot(X, losses_zero, color="r", label="0", linewidth=1)
    plt.plot(X, losses_point_five, color="g", label="0.5", linewidth=1)
    plt.plot(X, losses_one, color="b", label="1", linewidth=1)
    plt.ylabel("losses")
    plt.xlabel("iterations")
    plt.legend()
    plt.savefig("experiment15_losses.png")
