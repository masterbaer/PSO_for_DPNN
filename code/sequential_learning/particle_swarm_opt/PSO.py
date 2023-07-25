"""
This is our new PSO-implementation adapted for neural network training. As mentioned in our initial attempt
(particle_swarm_opt_initial_attmept/PSO) we change the following aspects:

1) we do not flatten the weights but instead work on parameters/state_dicts
2) we initialize particles with the default pytorch init and velocities with zero
3) We use small batches for fitness evaluation instead of the whole dataset.

"""

import copy
import time

from model import *
from helperfunctions import evaluate_model, evaluate_position_single_batch, reset_all_weights
from torch import nn




class Particle:

    def __init__(self, model):
        # initialize the particle randomly
        self.model = copy.deepcopy(model)
        reset_all_weights(self.model)

        # self.velocity = [(2 * torch.rand_like(param.data).to(device) - 1) for param in
        #                 self.model.parameters()]  # velocities in [-1,1]

        # initialize velocity with pytorch neural-net initialization instead of uniform[min,max]
        self.velocity = copy.deepcopy(model)
        reset_all_weights(self.velocity)

        self.best_model = copy.deepcopy(self.model)

        self.best_loss = float('inf')  # Change to the best accuracy/fitness?


class PSO:
    def __init__(self,
                 model,
                 num_particles: int,
                 inertia_weight: float,
                 social_weight: float,
                 cognitive_weight: float,
                 max_iterations: int,
                 train_loader,
                 valid_loader,
                 device):
        self.device = device

        self.model = model

        self.num_particles = num_particles

        self.particles = []
        for i in range(self.num_particles):
            self.particles.append(Particle(self.model))

        self.inertia_weight = torch.tensor(inertia_weight).to(self.device)
        self.social_weight = torch.tensor(social_weight).to(self.device)
        self.cognitive_weight = torch.tensor(cognitive_weight).to(self.device)

        self.max_iterations = max_iterations
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def optimize(self, evaluate=True, output1="particle_loss_list.pt", output2="particle_accuracy_list.pt"):
        train_generator = iter(self.train_loader)

        print("find initial global best model")
        try:
            train_inputs, train_labels = next(train_generator)
        except StopIteration:
            train_generator = iter(self.train_loader)
            train_inputs, train_labels = next(train_generator)
        train_inputs = train_inputs.to(self.device)
        train_labels = train_labels.to(self.device)

        for particle_index, particle in enumerate(self.particles):
            particle.model.to(self.device)
            particle_loss, particle_accuracy = evaluate_position_single_batch(particle.model, train_inputs,
                                                                              train_labels, self.device)
            particle.best_loss = particle_loss
            particle.model.to("cpu")

        global_best_loss = self.particles[0].best_loss
        global_best_model = copy.deepcopy(self.particles[0].model).to(self.device)
        for particle_index, particle in enumerate(self.particles):
            if particle.best_loss < global_best_loss:
                global_best_loss = particle.best_loss
                global_best_model = copy.deepcopy(particle.model).to(self.device)

        global_best_accuracy = 0.0
        particle_loss_list = torch.zeros(len(self.particles), self.max_iterations)
        particle_accuracy_list = torch.zeros(len(self.particles), self.max_iterations)

        #  Training Loop
        for iteration in range(self.max_iterations):
            try:
                train_inputs, train_labels = next(train_generator)
            except StopIteration:
                train_generator = iter(self.train_loader)
                train_inputs, train_labels = next(train_generator)
            train_inputs = train_inputs.to(self.device)
            train_labels = train_labels.to(self.device)

            for particle_index, particle in enumerate(self.particles):
                particle.model = particle.model.to(self.device)
                particle.best_model = particle.best_model.to(self.device)
                particle.velocity = particle.velocity.to(self.device)

                with torch.no_grad():
                    # Update particle velocity and position.

                    for i, (param_current, g_best, p_best, velocity_current) in enumerate(
                            zip(particle.model.parameters(), global_best_model.parameters(),
                                particle.best_model.parameters(), particle.velocity.parameters())):
                        social_component = self.social_weight * torch.rand(1).to(self.device) * (
                                g_best.data - param_current.data)
                        cognitive_component = self.cognitive_weight * torch.rand(1).to(self.device) * (
                                p_best.data - param_current.data)

                        velocity = velocity_current.data * self.inertia_weight + social_component + cognitive_component
                        # - param_current.grad.data * learning_rate
                        velocity_current.data = velocity
                        param_current.add_(velocity)

                # Evaluate particle fitness using the fitness function

                # particle_loss, particle_accuracy = evaluate_position(particle.model, self.valid_loader, self.device)
                particle_loss, particle_accuracy = evaluate_position_single_batch(particle.model, train_inputs,
                                                                                  train_labels, self.device)

                if evaluate:
                    particle_loss_list[particle_index, iteration] = particle_loss
                    particle_accuracy_list[particle_index, iteration] = particle_accuracy

                if iteration % 20 == 0:
                    print(f"(particle,loss,accuracy) = "
                          f"{(particle_index + 1, round(particle_loss, 3), round(particle_accuracy, 3))}")

                # Update particle's best position and fitness
                if particle_loss < particle.best_loss:
                    particle.best_loss = particle_loss
                    particle.best_model = copy.deepcopy(particle.model)

                # Update global best position and fitness
                if particle_loss < global_best_loss:
                    global_best_loss = particle_loss
                    global_best_model = copy.deepcopy(particle.model).to(self.device)

                if particle_accuracy > global_best_accuracy:
                    global_best_accuracy = particle_accuracy

                particle.model = particle.model.to("cpu")
                particle.best_model = particle.best_model.to("cpu")
                particle.velocity = particle.velocity.to("cpu")

            if iteration % 20 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Loss: {global_best_loss}")

        if evaluate:
            torch.save(particle_loss_list, output1)
            torch.save(particle_accuracy_list, output2)

        # overwrite the initial models parameters
        self.model.load_state_dict(global_best_model.state_dict())

        return global_best_loss, global_best_accuracy
