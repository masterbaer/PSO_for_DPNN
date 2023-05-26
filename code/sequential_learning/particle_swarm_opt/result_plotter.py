from matplotlib import pyplot as plt
import matplotlib
from celluloid import Camera
from matplotlib import cm

import torch
import numpy as np


def print_losses():
    losses = torch.load('particle_loss_list.pt')
    particle_size, iterations = losses.shape

    x = list(range(1, iterations))

    for i in range(particle_size):
        plt.plot(x, losses[i][x], label=f"Particle {i + 1}")

    plt.legend(loc='best')
    plt.ylabel("losses")
    plt.show()
    plt.cla()


def get_global_losses():
    losses = torch.load('particle_loss_list.pt')
    particle_size, iterations = losses.shape

    # print the best losses until that iteration
    gbest_list = torch.full((iterations,), float('inf'))

    gbest_list[0] = torch.min(losses[:, 0])
    for i in range(iterations - 1):
        new_best_loss = torch.min(losses[:, i + 1])
        gbest_list[i + 1] = torch.min(gbest_list[i], new_best_loss)
    return gbest_list


def get_global_accuracies():
    particles = torch.load('pca_weights.pt')  # (iteration, particle, 2)
    losses = torch.load("particle_loss_list.pt")  # (particle,iteration)
    accuracies = torch.load("particle_accuracy_list.pt")
    number_of_iterations, number_of_particles, number_of_pca_components = particles.shape

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


def print_global_loss():
    gbest_list = get_global_losses()
    plt.plot(gbest_list)
    plt.ylabel("global best loss")
    plt.show()
    plt.cla()


def print_global_accuracy():
    best_accuracies = get_global_accuracies()
    plt.plot(best_accuracies)
    plt.ylabel("global accuracy")
    plt.show()
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


""""
def print_pca():
    particles = torch.load('pca_weights.pt')  # (iteration, particle, 2)
    losses = torch.load("particle_loss_list.pt")  # (particle,iteration)
    accuracies = torch.load("particle_accuracy_list.pt")
    number_of_iterations, number_of_particles, number_of_pca_components = particles.shape

    min_loss = torch.min(losses[:, :])
    min_loss = min_loss.item()
    max_loss = torch.max(losses[:, :]).item()
    #  max_accuracy = torch.max(accuracies).item()

    best_index = torch.argmin(losses)
    best_particle = best_index // number_of_iterations
    best_iteration = best_index % number_of_iterations

    sizes = 50 * (max_loss - losses) / (max_loss - min_loss)  # normalized fitness

    print(number_of_iterations, "iterations, ", number_of_particles, "particles , ", number_of_pca_components,
          "pca components")

    color_indices = np.linspace(0.0, 1.0, number_of_particles)
    colors = matplotlib.colormaps["viridis"]

    show_particles = list(range(number_of_particles))

    for particle_index in range(number_of_particles):
        color = colors(color_indices[particle_index])
        if particle_index in show_particles:
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
        if particle_index in show_particles:
            for iteration in range(number_of_iterations - 1):
                x0 = particles[iteration, particle_index, 0]
                y0 = particles[iteration, particle_index, 1]
                x1 = particles[iteration + 1, particle_index, 0]
                y1 = particles[iteration + 1, particle_index, 1]
                plt.arrow(x0, y0, x1 - x0, y1 - y0, color=color, head_length=0.3, head_width=0.05)

    plt.show()
    # plt.savefig('pso_sequential_experiment.png')

    plt.cla()
"""


def create_animation():
    # See https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot

    particles = torch.load('pca_weights.pt')  # (iteration, particle, 2)
    losses = torch.load("particle_loss_list.pt")  # (particle,iteration)
    accuracies = torch.load("particle_accuracy_list.pt")
    number_of_iterations, number_of_particles, number_of_pca_components = particles.shape

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

    # currently plotted points and gbest
    points = np.random.random((2, number_of_particles))

    color_indices = np.linspace(0.0, 1.0, number_of_particles)
    colors = matplotlib.colormaps['rainbow'](color_indices)

    fig, ax = plt.subplots()
    camera = Camera(fig)

    last_relevant_iteration = 0
    for iteration in range(number_of_iterations):
        # to give the accuracy from the iteration where the best loss was measured, we have to save the iteration as well
        if gbest_losses[iteration] < gbest_losses[last_relevant_iteration]:
            last_relevant_iteration = iteration

        for particle_index in range(number_of_particles):
            # Update points
            points[:, particle_index] = particles[iteration, particle_index]

        plt.title("Particle Swarm Optimization")
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')

        for particle_index in range(number_of_particles):
            plt.text(particles[iteration, particle_index][0], particles[iteration, particle_index][1],
                     f"p{particle_index + 1}")

        best_loss = gbest_losses[iteration]
        best_particle = gbest_particles[iteration]
        best_weight = particles[last_relevant_iteration, best_particle]

        plt.text(0.01, 0.99, f'iteration {iteration}', ha='left', va='top', transform=ax.transAxes)
        plt.text(0.01, 0.95, f'best loss:  {round(best_loss.item(), 3)}', ha='left', va='top', transform=ax.transAxes)
        plt.text(0.01, 0.91, f'best accuracy:  {round(accuracies[best_particle, last_relevant_iteration].item(), 3)}',
                 ha='left', va='top', transform=ax.transAxes)

        #  plot all points
        plt.scatter(*points, c=colors, s=25)

        # plot gbest
        gbest = np.random.random((2, 1))
        gbest[:, 0] = best_weight
        plt.scatter(*gbest, c="black", s=50, marker="X")

        camera.snap()

        if iteration == 0:
            plt.savefig("plot_at_beginning.png")
        if iteration == number_of_iterations - 1:
            anim = camera.animate(blit=True, interval=700)
            anim.save('scatter.mp4')
            plt.cla()

            # Plot the particles in the last iteration
            fig, ax = plt.subplots()
            plt.title("Particle Swarm Optimization")
            plt.xlabel('PCA1')
            plt.ylabel('PCA2')
            plt.scatter(*points, c=colors, s=25)
            plt.scatter(*gbest, c="black", s=50, marker="X")
            for particle_index in range(number_of_particles):
                plt.text(particles[iteration, particle_index][0], particles[iteration, particle_index][1],
                         f"p{particle_index + 1}")
            plt.text(0.01, 0.99, f'iteration {iteration}', ha='left', va='top', transform=ax.transAxes)
            plt.text(0.01, 0.95, f'best loss:  {round(best_loss.item(), 3)}', ha='left', va='top',
                     transform=ax.transAxes)
            plt.text(0.01, 0.91,
                     f'best accuracy:  {round(accuracies[best_particle, last_relevant_iteration].item(), 3)}',
                     ha='left', va='top', transform=ax.transAxes)
            plt.savefig("plot_at_last_iteration.png")





if __name__ == '__main__':
    valid_losses = get_global_losses()
    valid_accuracies = get_global_accuracies()

    plt.plot(valid_losses)
    plt.ylabel("validation loss")
    plt.savefig('seq_pso_valid_loss.png')
    plt.cla()

    plt.plot(valid_accuracies)
    plt.ylabel("validation accuracy")
    plt.savefig('seq_pso_valid_acc.png')
    plt.cla()

    create_animation()
