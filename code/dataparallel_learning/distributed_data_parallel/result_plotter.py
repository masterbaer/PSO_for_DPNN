from matplotlib import pyplot as plt
import torch


def save_loss(loss_list):
    plt.plot(loss_list)
    plt.ylabel("loss")
    plt.savefig('ddp_loss.png')
    plt.cla()
    return


def save_train_acc(train_acc_list):
    plt.plot(train_acc_list)
    plt.ylabel("training accuracy")
    plt.savefig('ddp_train_acc.png')
    plt.cla()
    return


def save_valid_acc(valid_acc_list):
    plt.plot(valid_acc_list)
    plt.ylabel("validation accuracy")
    plt.savefig('ddp_valid_acc.png')
    plt.cla()
    return


if __name__ == '__main__':
    loss_list = torch.load("ddp_loss.pt")
    train_acc_list = torch.load("ddp_train_acc.pt")
    valid_acc_list = torch.load("ddp_valid_acc.pt")

    plt.plot(loss_list)
    plt.ylabel("loss")
    plt.savefig('ddp_loss.png')
    plt.cla()

    plt.plot(train_acc_list)
    plt.ylabel("training accuracy")
    plt.savefig('ddp_train_acc.png')
    plt.cla()

    plt.plot(valid_acc_list)
    plt.ylabel("validation accuracy")
    plt.savefig('ddp_valid_acc.png')
    plt.cla()
