from matplotlib import pyplot as plt, colormaps
import torch
import numpy as np
import networkx as nx


def print_loss():
    losses = torch.load('particle_loss_list.pt')
    particle_size, iterations = losses.shape

    x = list(range(1, iterations))

    for i in range(particle_size):
        plt.plot(x, losses[i][x], label=f"Particle {i + 1}")

    plt.legend(loc='best')
    plt.ylabel("losses")
    plt.show()
    plt.cla()


def print_accuracy():
    accuracies = torch.load('particle_accuracy_list.pt')
    particle_size, iterations = accuracies.shape

    x = list(range(1, iterations))

    for i in range(particle_size):
        plt.plot(x, accuracies[i][x], label=f"Particle {i + 1}")

    plt.legend(loc='best')
    plt.ylabel("accuracies")
    plt.show()
    plt.cla()


def print_distances():
    a = np.array([(0, 0.3, 0.4, 0.7),
                  (0.3, 0, 0.9, 0.2),
                  (0.4, 0.9, 0, 0.1),
                  (0.7, 0.2, 0.1, 0)
                  ])
    G = nx.from_numpy_array(a)
    nx.draw(G, edge_color=[i[2]['weight'] for i in G.edges(data=True)], edge_cmap=colormaps['winter'])
    plt.savefig("distances.png")
    plt.cla()


def print_pca():
    pass
    #   TODO


if __name__ == '__main__':
    print_distances()
