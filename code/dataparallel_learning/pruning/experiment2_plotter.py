import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':

    accuracies_combined = torch.load("ex2_combined_test_accuracy.pt")
    accuracies_combined_scratch = torch.load("ex2_combined_scratch_test_accuracy.pt")

    plt.plot(accuracies_combined, label="combined model")
    plt.plot(accuracies_combined_scratch, label="big model from scratch")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracies for finetuning")
    plt.savefig("ex2_accuracies.png")
    plt.cla()
