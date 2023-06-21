

import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    losses = torch.load("global_loss_list.pt")  # (particle,iteration)
    accuracies = torch.load("global_accuracy_list.pt")

    plt.plot(losses)
    plt.ylabel("losses")
    plt.savefig("parallel_losses_double_batch.png")
    #plt.show()
    plt.cla()
