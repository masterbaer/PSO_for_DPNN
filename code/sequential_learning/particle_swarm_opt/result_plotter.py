from matplotlib import pyplot as plt
import torch
import numpy as np

if __name__ == '__main__':
    losses = torch.load('particle_loss_list.pt')
    particle_size, iterations = losses.shape

    x = list(range(1, iterations))

    for i in range(particle_size):
        plt.plot(x, losses[i][x], label=f"Particle {i+1}")

    plt.legend(loc='best')
    # plt.plot(losses) # first particle
    plt.ylabel("losses")
    plt.show()

    # plt.savefig('seq_gd_valid_acc.png')
    plt.cla()


