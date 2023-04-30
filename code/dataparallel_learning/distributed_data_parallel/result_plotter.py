import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    loss = torch.load('ddp_loss.pt')
    plt.plot(loss)
    plt.ylabel("loss ddp")
    plt.show()

    train_acc = torch.load('ddp_train_acc.pt')
    plt.plot(train_acc)
    plt.ylabel("training accuracy ddp")
    plt.show()

    valid_acc = torch.load('ddp_valid_acc.pt')
    plt.plot(valid_acc)
    plt.ylabel("validation accuracy ddp")
    plt.show()
