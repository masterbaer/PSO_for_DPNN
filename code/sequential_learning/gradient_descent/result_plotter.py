import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    loss = torch.load('seq_gd_loss.pt')
    plt.plot(loss)
    plt.ylabel("loss gd")
    plt.savefig('seq_gd_loss.png')

    train_acc = torch.load('seq_gd_train_acc.pt')
    plt.plot(train_acc)
    plt.ylabel("training accuracy gd")
    plt.savefig('seq_gd_train_acc.png')

    valid_acc = torch.load('seq_gd_valid_acc.pt')
    plt.plot(valid_acc)
    plt.ylabel("validation accuracy gd")
    plt.savefig('seq_gd_valid_acc.png')
