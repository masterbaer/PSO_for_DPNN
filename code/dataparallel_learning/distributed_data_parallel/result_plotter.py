from matplotlib import pyplot as plt


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
