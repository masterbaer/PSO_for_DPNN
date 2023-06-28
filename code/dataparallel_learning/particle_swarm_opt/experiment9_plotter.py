

import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    losses = torch.load("experiment9_loss.pt")  # (particle,iteration)
    accuracies = torch.load("experiment9_accuracy.pt")

    plt.plot(losses)
    plt.ylabel("losses")
    plt.savefig("parallel_losses_double_batch.png")
    #plt.show()
    plt.cla()

    plt.plot(accuracies)
    plt.ylabel("accuracies")
    plt.savefig("experiment9_accuracy.png")
    #plt.show()
    plt.cla()
