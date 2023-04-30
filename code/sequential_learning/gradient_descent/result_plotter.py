from matplotlib import pyplot as plt


def save_plots(loss_list, train_acc_list, valid_acc_list):
    plt.plot(loss_list)
    plt.ylabel("loss gd")
    plt.savefig('seq_gd_loss.png')

    plt.plot(train_acc_list)
    plt.ylabel("training accuracy gd")
    plt.savefig('seq_gd_train_acc.png')

    plt.plot(valid_acc_list)
    plt.ylabel("validation accuracy gd")
    plt.savefig('seq_gd_valid_acc.png')
