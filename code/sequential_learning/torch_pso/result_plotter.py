from matplotlib import pyplot as plt
import torch


def save_loss(loss_list):
    plt.plot(loss_list)
    plt.ylabel("loss")
    plt.savefig('torch_pso_loss.png')
    plt.cla()
    return


def save_train_acc(train_acc_list):
    plt.plot(train_acc_list)
    plt.ylabel("training accuracy")
    plt.savefig('torch_pso_train_acc.png')
    plt.cla()
    return


def save_valid_acc(valid_acc_list):
    plt.plot(valid_acc_list)
    plt.ylabel("validation accuracy")
    plt.savefig('torch_pso_valid_acc.png')
    plt.cla()
    return
