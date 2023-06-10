from matplotlib import pyplot as plt
import matplotlib
from celluloid import Camera
from matplotlib import cm

import torch
import numpy as np


def get_global_accuracies():
    losses = torch.load("particle_loss_list.pt")  # (particle,iteration)
    accuracies = torch.load("particle_accuracy_list.pt")
    number_of_particles, number_of_iterations = losses.shape

    # get the global best loss in each iteration
    gbest_losses = torch.full((number_of_iterations,), float('inf'))
    gbest_particles = torch.full((number_of_iterations,), 0)

    gbest_losses[0] = torch.min(losses[:, 0])
    gbest_particles[0] = torch.argmin(losses[:, 0])

    for i in range(number_of_iterations - 1):
        new_best_loss = torch.min(losses[:, i + 1])
        particle_index = torch.argmin(losses[:, i + 1])
        if new_best_loss < gbest_losses[i]:
            gbest_losses[i + 1] = new_best_loss
            gbest_particles[i + 1] = particle_index
        else:
            gbest_losses[i + 1] = gbest_losses[i]
            gbest_particles[i + 1] = gbest_particles[i]

    best_accuracies = torch.zeros(number_of_iterations)
    last_relevant_iteration = 0
    for iteration in range(number_of_iterations):
        if gbest_losses[iteration] < gbest_losses[last_relevant_iteration]:
            last_relevant_iteration = iteration

        best_particle = gbest_particles[iteration]
        best_accuracies[iteration] = accuracies[best_particle, last_relevant_iteration]
    return best_accuracies


def print_global_accuracy():
    best_accuracies = get_global_accuracies()
    plt.plot(best_accuracies)
    plt.ylabel("global accuracy")

    plt.savefig("global_accuracy_pso_with_gradients.png")
    # plt.show()
    plt.cla()


def print_accuracies():
    accuracies = torch.load('particle_accuracy_list.pt')
    particle_size, iterations = accuracies.shape

    x = list(range(1, iterations))

    for i in range(particle_size):
        plt.plot(x, accuracies[i][x], label=f"Particle {i + 1}")

    plt.legend(loc='best')
    plt.ylabel("accuracies")
    plt.show()
    plt.cla()


if __name__ == '__main__':
    print_global_accuracy()
