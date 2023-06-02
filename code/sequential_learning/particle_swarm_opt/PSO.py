import time

import numpy as np
import torch
from sklearn.decomposition import PCA
import copy
from model import *

from torch import nn


def reset_all_weights(model: nn.Module) -> None:
    """
    copied from: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/11

    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


# The particle evaluation is similar to the validation process.
# see "Per-Epoch Activity" https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def evaluate_position(model, data_loader, device):
    #  Compute Loss
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    valid_loss = 0.0
    number_of_batches = len(data_loader)
    correct_pred, num_examples = 0, 0

    for i, (vinputs, vlabels) in enumerate(data_loader):  # Loop over batches in data.
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        predictions = model(vinputs)  # Calculate model output.
        _, predicted = torch.max(predictions, dim=1)  # Determine class with max. probability for each sample.
        num_examples += vlabels.size(0)  # Update overall number of considered samples.
        correct_pred += (predicted == vlabels).sum()  # Update overall number of correct predictions.
        loss = loss_fn(predictions, vlabels)
        # loss = torch.nn.functional.cross_entropy(predictions, vlabels)  # cross-entropy loss for multiclass

        valid_loss = valid_loss + loss.item()

    loss_per_batch = valid_loss / number_of_batches
    accuracy = (correct_pred.float() / num_examples).item()
    return loss_per_batch, accuracy


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
        self.valid_loader = valid_loader

    def optimize(self, evaluate=True):

        print("initial evaluation")
        with torch.no_grad():
            for particle_index, particle in enumerate(self.particles):
                particle_loss, particle_accuracy = evaluate_position(particle.model, self.valid_loader, self.device)
                print("particle,loss,accuracy = ",
                      (particle_index + 1, round(particle_loss, 3), round(particle_accuracy, 5)))

        # num_weights = sum(p.numel() for p in self.particles[0].model.parameters())

        global_best_loss = float('inf')
        global_best_accuracy = 0.0
        global_best_model = copy.deepcopy(self.particles[0].model)  # Init as the first model
        global_best_model = global_best_model.to(self.device)

        particle_loss_list = torch.zeros(len(self.particles), self.max_iterations)
        particle_accuracy_list = torch.zeros(len(self.particles), self.max_iterations)

        #  Training Loop
        for iteration in range(self.max_iterations):
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
                particle_loss, particle_accuracy = evaluate_position(particle.model, self.valid_loader, self.device)
                if evaluate:
                    particle_loss_list[particle_index, iteration] = particle_loss
                    particle_accuracy_list[particle_index, iteration] = particle_accuracy

                # if iteration % 20 == 0:
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

            # if iteration % 20 == 0:
            print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Loss: {global_best_loss}")

        # TODO timestamp/hyperparameter/modell in Namen reintun
        if evaluate:
            torch.save(particle_loss_list, "particle_loss_list.pt")
            torch.save(particle_accuracy_list, "particle_accuracy_list.pt")

        # overwrite the initial models parameters
        self.model.load_state_dict(global_best_model.state_dict())

        return global_best_loss, global_best_accuracy
