

import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    losses = torch.load("global_loss_list.pt")  # (particle,iteration)
    accuracies = torch.load("global_accuracy_list.pt")

    plt.plot(accuracies)
    plt.ylabel("accuracies")
    plt.show()
    plt.cla()
