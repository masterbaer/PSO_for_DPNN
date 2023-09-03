import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':

    accuracies_model0 = torch.load("ex7_test_accuracy_0.pt")
    times_model0 = torch.load("ex7_cumulative_times.pt")
    accuracies_sgd = torch.load("ex7b_test_accuracy_0.pt")
    times_sgd = torch.load("ex7b_cumulative_times.pt")

    plt.plot(times_model0, accuracies_model0 , label="parallel pruning")
    plt.plot(times_sgd, accuracies_sgd, label="sgd")

    plt.legend()
    plt.xlabel("time")
    plt.ylabel("accuracies")
    plt.savefig("ex7_accuracies.png")
    plt.cla()
