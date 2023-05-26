import torch
from matplotlib import pyplot as plt


def save_loss(loss_list):
    plt.plot(loss_list)
    plt.ylabel("loss")
    plt.savefig('seq_gd_loss.png')
    plt.cla()
    return


def save_train_acc(train_acc_list):
    plt.plot(train_acc_list)
    plt.ylabel("training accuracy")
    plt.savefig('seq_gd_train_acc.png')
    plt.cla()
    return


def save_valid_acc(valid_acc_list):
    plt.plot(valid_acc_list)
    plt.ylabel("validation accuracy")
    plt.savefig('seq_gd_valid_acc.png')
    plt.cla()
    return

if __name__ == '__main__':
    losses = torch.load("seq_gd_loss.pt")
    accuracies_train = torch.load("seq_gd_train_acc.pt")
    accuracies_valid = torch.load("seq_gd_valid_acc.pt")

