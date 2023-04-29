import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    loss = torch.load('output/loss.pt')
    plt.plot(loss)
    plt.show()

    train_acc = torch.load('output/train_acc.pt')
    plt.plot(train_acc)
    plt.show()

    valid_acc = torch.load('output/valid_acc.pt')
    plt.plot(valid_acc)
    plt.show()
