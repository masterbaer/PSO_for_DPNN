import time
from typing import Any, Dict, Callable, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
import copy

from torch.optim import Optimizer

from model import *

from torch import nn

# TODO implement this class when absolutely necessary.


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


class Particle:

    def __init__(self, model): # TODO adapt to parameters
        # initialize the particle randomly
        self.model = copy.deepcopy(model) # TODO adapt to parameters
        reset_all_weights(self.model) # TODO adapt to parameters

        self.velocity = copy.deepcopy(model) # TODO adapt to parameters
        reset_all_weights(self.velocity) # TODO adapt to parameters

        self.best_model = copy.deepcopy(self.model) # TODO adapt to parameters

        self.best_loss = float('inf')  # Change to the best accuracy/fitness?


class PSO(Optimizer):
    def __init__(self, params, num_particles: int, inertia_weight: float, social_weight: float, cognitive_weight: float,
                 max_iterations: int, valid_loader, device, defaults: Dict[str, Any]):
        super().__init__(params, defaults)
        self.device = device

        self.params = params

        self.num_particles = num_particles

        self.particles = []
        for i in range(self.num_particles):
            self.particles.append(Particle(self.params))

        self.inertia_weight = torch.tensor(inertia_weight).to(self.device)
        self.social_weight = torch.tensor(social_weight).to(self.device)
        self.cognitive_weight = torch.tensor(cognitive_weight).to(self.device)

        self.max_iterations = max_iterations

    def step(self, closure: Optional[Callable[[], float]] = None, evaluate=True) -> float:

        global_best_loss = float('inf')
        global_best_accuracy = 0.0
        global_best_model = copy.deepcopy(self.particles[0].model)  # Init as the first model # TODO adapt to parameters
        global_best_model = global_best_model.to(self.device) # TODO adapt to parameters

        particle_loss_list = torch.zeros(len(self.particles), self.max_iterations)
        particle_accuracy_list = torch.zeros(len(self.particles), self.max_iterations)

        #  Training Loop
        for iteration in range(self.max_iterations):
            for particle_index, particle in enumerate(self.particles):

                particle.model = particle.model.to(self.device) # TODO adapt to parameters
                particle.best_model = particle.best_model.to(self.device) # TODO adapt to parameters
                particle.velocity = particle.velocity.to(self.device) # TODO adapt to parameters

                with torch.no_grad():
                    # Update particle velocity and position.

                    for i, (param_current, g_best, p_best, velocity_current) in enumerate( # TODO adapt to parameters
                            zip(particle.model.parameters(), global_best_model.parameters(), # TODO adapt to parameters
                                particle.best_model.parameters(), particle.velocity.parameters())): # TODO adapt to parameters
                        social_component = self.social_weight * torch.rand(1).to(self.device) * (
                                g_best.data - param_current.data)
                        cognitive_component = self.cognitive_weight * torch.rand(1).to(self.device) * (
                                p_best.data - param_current.data)

                        velocity = velocity_current * self.inertia_weight + social_component + cognitive_component
                        # - param_current.grad.data * learning_rate

                        param_current.add_(velocity)


                # Evaluate particle fitness using the fitness function
                # TODO assign the parameters to global model
                particle_loss, particle_accuracy = closure()

                if evaluate:
                    particle_loss_list[particle_index, iteration] = particle_loss
                    particle_accuracy_list[particle_index, iteration] = particle_accuracy

                # if iteration % 20 == 0:
                print(f"(particle,loss,accuracy) = "
                      f"{(particle_index + 1, round(particle_loss, 3), round(particle_accuracy, 3))}")

                # Update particle's best position and fitness
                if particle_loss < particle.best_loss:
                    particle.best_loss = particle_loss
                    particle.best_model = copy.deepcopy(particle.model) # TODO adapt to parameters

                # Update global best position and fitness
                if particle_loss < global_best_loss:
                    global_best_loss = particle_loss
                    global_best_model = copy.deepcopy(particle.model).to(self.device)

                if particle_accuracy > global_best_accuracy:
                    global_best_accuracy = particle_accuracy

                particle.model = particle.model.to("cpu")
                particle.best_model = particle.best_model.to("cpu")
                particle.velocity = particle.velocity.to("cpu")

                # end6 = time.time()
                # print("updating gbest and converting to cpu took: ", end6 - start6, "seconds ")

            # if iteration % 20 == 0:
            print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Loss: {global_best_loss}")

        # TODO timestamp/hyperparameter/modell in Namen reintun
        if evaluate:
            torch.save(particle_loss_list, "particle_loss_list.pt")
            torch.save(particle_accuracy_list, "particle_accuracy_list.pt")

        # overwrite the initial models parameters
        self.model.load_state_dict(global_best_model.state_dict())

        return global_best_loss