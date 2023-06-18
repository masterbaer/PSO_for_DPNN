import copy
import time

from model import *
from helperfunctions import evaluate_model, evaluate_position_single_batch, reset_all_weights
from torch import nn


# Benchmark: Using normal SGD, 175 batches per epoch, ReduceLROnPlateau: factor=0.1, mode='max', verbose=True
#                                                                            Loss: 2.3024 (on first training batch)
# Epoch: 001/040 | Train: 0.43 | Validation: 0.41 , Time elapsed: 0.80 min , Loss: 1.7122
# Epoch: 002/040 | Train: 0.47 | Validation: 0.44 , Time elapsed: 1.64 min
# Epoch: 003/040 | Train: 0.50 | Validation: 0.48  , Time elapsed: 2.45 min
# Epoch: 004/040 | Train: 0.53 | Validation: 0.49, Time elapsed: 3.29 min
# Epoch: 005/040 | Train: 0.54 | Validation: 0.50, Time elapsed: 4.11 min
# Epoch: 006/040 | Train: 0.56 | Validation: 0.51 , Time elapsed: 4.91 min
# Epoch: 007/040 | Train: 0.58 | Validation: 0.52 , Time elapsed: 5.71 min
# Epoch: 008/040 | Train: 0.59 | Validation: 0.53 , Time elapsed: 6.51 min
# Epoch: 011/040 | Train: 0.62 | Validation: 0.54 , Time elapsed: 8.83 min , Loss 1.1689
# Epoch: 014/040 | Train: 0.64 | Validation: 0.55 , Time elapsed: 11.23 min, Loss: 1.2129
# Epoch: 017/040 | Train: 0.68 | Validation: 0.56 , Time elapsed: 13.59 min, Loss: 1.1200
# Epoch: 020/040 | Train: 0.70 | Validation: 0.57 , Time elapsed: 15.93 min , Loss: 1.0405
# Epoch: 031/040 | Train: 0.77 | Validation: 0.57, Time elapsed: 24.45 min , Loss: 0.7542
# Epoch: 040/040 | Train: 0.82 | Validation: 0.57, Time elapsed: 31.52 min , Loss: 0.4544 (test accuracy 57%)

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

    def optimize(self, evaluate=True):

        print("initial evaluation")
        with torch.no_grad():
            for particle_index, particle in enumerate(self.particles):
                particle_loss, particle_accuracy = evaluate_model(particle.model, self.valid_loader, self.device)
                print("particle,loss,accuracy = ",
                      (particle_index + 1, round(particle_loss, 3), round(particle_accuracy, 5)))
        # num_weights = sum(p.numel() for p in self.particles[0].model.parameters())

        global_best_loss = float('inf')
        global_best_accuracy = 0.0
        global_best_model = copy.deepcopy(self.particles[0].model)  # Init as the first model
        global_best_model = global_best_model.to(self.device)

        particle_loss_list = torch.zeros(len(self.particles), self.max_iterations)
        particle_accuracy_list = torch.zeros(len(self.particles), self.max_iterations)

        train_generator = iter(self.train_loader)
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

                        velocity = velocity_current * self.inertia_weight + social_component + cognitive_component
                        # - param_current.grad.data * learning_rate

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

        # TODO timestamp/hyperparameter/modell in Namen reintun
        if evaluate:
            torch.save(particle_loss_list, "particle_loss_list.pt")
            torch.save(particle_accuracy_list, "particle_accuracy_list.pt")

        # overwrite the initial models parameters
        self.model.load_state_dict(global_best_model.state_dict())

        return global_best_loss, global_best_accuracy
