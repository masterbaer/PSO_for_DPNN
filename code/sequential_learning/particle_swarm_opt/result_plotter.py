from matplotlib import pyplot as plt
import matplotlib

import torch
import numpy as np


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


def print_pca():
    particles = torch.load('pca_weights.pt')  # (iteration, particle, 2)
    losses = torch.load("particle_loss_list.pt")  # (particle,iteration)
    accuracies = torch.load("particle_accuracy_list.pt")
    number_of_iterations, number_of_particles, number_of_pca_components = particles.shape

    min_loss = torch.min(losses[:, :])
    min_loss = min_loss.item()
    max_loss = torch.max(losses[:, :]).item()
    max_accuracy = torch.max(accuracies).item()

    best_index = torch.argmin(losses)
    best_particle = best_index // number_of_iterations
    best_iteration = best_index % number_of_iterations

    sizes = 50 * (max_loss - losses) / (max_loss - min_loss)  # normalized fitness

    print(number_of_iterations, "iterations, ", number_of_particles, "particles , ", number_of_pca_components,
          "pca components")

    print(accuracies)

    color_indices = np.linspace(0.0, 1.0, number_of_particles)
    colors = matplotlib.colormaps["viridis"]

    for particle_index in range(number_of_particles):
        color = colors(color_indices[particle_index])
        x = particles[:, particle_index, 0]
        y = particles[:, particle_index, 1]
        plt.scatter(x, y, color=color, s=sizes[particle_index, :])

        plt.text(x[0], y[0], f"particle {particle_index + 1}", color=color)

    plt.text(particles[best_iteration, best_particle][0],
             particles[best_iteration, best_particle][1],
             f"gbest")

    # Make arrows
    for particle_index in range(number_of_particles):
        color = colors(color_indices[particle_index])
        for iteration in range(number_of_iterations - 1):
            x0 = particles[iteration, particle_index, 0]
            y0 = particles[iteration, particle_index, 1]
            x1 = particles[iteration + 1, particle_index, 0]
            y1 = particles[iteration + 1, particle_index, 1]
            plt.arrow(x0, y0, x1 - x0, y1 - y0, color=color, head_length=0.3, head_width=0.05)
    plt.show()
    plt.cla()


if __name__ == '__main__':
    print_pca()
