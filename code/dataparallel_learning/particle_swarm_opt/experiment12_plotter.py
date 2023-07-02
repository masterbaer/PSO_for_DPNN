import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    #losses = torch.load("experiment12_loss.pt")  # (particle,iteration)
    accuracies = torch.load("experiment12_accuracy.pt")

    plt.plot(accuracies)
    plt.ylabel("accuracies")
    plt.savefig("experiment12_accuracy.png")
    #plt.show()
    plt.cla()

   # plt.plot(losses)
   # plt.ylabel("losses")
   # plt.savefig("experiment12_loss.png")
    #plt.show()
    #plt.cla()