from matplotlib import pyplot as plt


def save_plots(loss_list, train_acc_list, valid_acc_list):
    plt.plot(loss_list)
    plt.ylabel("loss ddp")
    plt.show()

    plt.plot(train_acc_list)
    plt.ylabel("training accuracy ddp")
    plt.show()

    plt.plot(valid_acc_list)
    plt.ylabel("validation accuracy ddp")
    plt.show()
