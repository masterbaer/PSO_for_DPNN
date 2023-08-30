import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':

    accuracies = torch.load("ex6_combined_test_accuracy_finetune.pt")
    plt.plot(accuracies, label="finetuning accuracy using the full dataset")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracies")
    plt.savefig("ex6_accuracies_finetune.png")
    plt.cla()
