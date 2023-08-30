import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':

    accuracies_model0 = torch.load("ex6_test_accuracy0.pt")
    accuracies_combined = torch.load("ex6_combined_test_accuracy.pt")
    accuracies_model0_full_dataset = torch.load("ex1_test_accuracy0.pt")
    accuracies_combined_full_dataset = torch.load("ex1_combined_test_accuracy.pt")


    plt.plot(accuracies_model0, label="model 0 data parallel")
    plt.plot(accuracies_model0_full_dataset, label="model 0 full dataset")
    plt.plot(accuracies_combined, label="combined model data parallel")
    plt.plot(accuracies_combined_full_dataset, label="combined model full dataset")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracies")
    plt.savefig("ex6_accuracies.png")
    plt.cla()
