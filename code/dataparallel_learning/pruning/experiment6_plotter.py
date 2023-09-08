import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':

    accuracies_model0 = torch.load("ex6_test_accuracy_dataparallel_0.pt")
    accuracies_combined = torch.load("ex6_combined_dataparallel_test_accuracy.pt")
    accuracies_model0_full_dataset = torch.load("ex6_test_accuracy_full_0.pt")
    accuracies_combined_full_dataset = torch.load("ex6_combined_full_test_accuracy.pt")
    times_dataparallel = torch.load("ex6_time_list_dataparallel.pt")
    times_full = torch.load("ex6_time_list_full.pt")

    plt.plot(accuracies_model0, label="model 0 data parallel")
    plt.plot(accuracies_model0_full_dataset, label="model 0 full dataset")
    plt.plot(accuracies_combined, label="combined model data parallel")
    plt.plot(accuracies_combined_full_dataset, label="combined model full dataset")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracies")
    plt.savefig("ex6_accuracies_epochs.png")
    plt.cla()



    plt.plot(times_dataparallel, accuracies_model0, label="model 0 data parallel")
    plt.plot(times_full, accuracies_model0_full_dataset, label="model 0 full dataset")
    plt.plot(times_dataparallel, accuracies_combined, label="combined model data parallel")
    plt.plot(times_full, accuracies_combined_full_dataset, label="combined model full dataset")

    plt.legend()
    plt.xlabel("seconds")
    plt.ylabel("accuracies")
    plt.savefig("ex6_accuracies_seconds.png")
    plt.cla()

