import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    losses = torch.load("experiment8_loss.pt")  # (particle,iteration)
    accuracies = torch.load("experiment8_accuracy.pt")

    plt.plot(accuracies)
    plt.ylabel("accuracies")
    plt.savefig("experiment8_accuracy.png")
    #plt.show()
    plt.cla()

    plt.plot(losses)
    plt.ylabel("losses")
    plt.savefig("experiment8_loss.png")
    #plt.show()
    plt.cla()
