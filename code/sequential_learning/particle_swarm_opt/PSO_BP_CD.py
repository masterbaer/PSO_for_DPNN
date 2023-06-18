import numpy as np
import torch
from sklearn.decomposition import PCA
import copy
from model import *

from torch import nn
from helperfunctions import evaluate_position_single_batch, reset_all_weights


# This optimizer is from PSO-PINN. https://arxiv.org/pdf/2202.01943.pdf
# The gradient is used as another velocity component and the social and cognitivce coefficients
# decay with 1/n where n is the number of iterations.

# In contrast to the paper we also let the inertia decay since otherwise the loss explodes (with given inertia).
# At some point, however, this becomes normal SGD with many neural networks in parallel.

class Particle:

    def __init__(self, model):
        # maybe only save the parameters?

        # initialize the particle randomly
        self.model = copy.deepcopy(model)
        reset_all_weights(self.model)

        # self.velocity = copy.deepcopy(model)
        # reset_all_weights(self.velocity)

        # initialize parameters as zero
        self.velocity = copy.deepcopy(model)
        for param in self.velocity.parameters():
            param.data = torch.zeros_like(param.data)
            # or use param.data.fill_(0)

        self.best_model = copy.deepcopy(self.model)

        self.best_loss = float('inf')  # Change to the best accuracy/fitness?


class PSO_BP_CD:
    def __init__(self,
                 model,
                 num_particles: int,
                 inertia_weight: float,
                 social_weight: float,
                 cognitive_weight: float,
                 max_iterations: int,
                 learning_rate: float,
                 valid_loader,
                 train_loader,
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

        self.learning_rate = torch.tensor(learning_rate).to(self.device)
        self.max_iterations = max_iterations
        self.valid_loader = valid_loader
        self.train_loader = train_loader  # for gradient calculation

    def optimize(self, evaluate=True):

        # print("initial evaluation")
        # with torch.no_grad():
        #    for particle_index, particle in enumerate(self.particles):
        #        particle_loss, particle_accuracy = evaluate_position(particle.model, self.valid_loader, self.device)
        #        print("particle,loss,accuracy = ",
        #              (particle_index + 1, round(particle_loss, 3), round(particle_accuracy, 5)))

        # num_weights = sum(p.numel() for p in self.particles[0].model.parameters())

        global_best_loss = float('inf')
        global_best_accuracy = 0.0
        global_best_model = copy.deepcopy(self.particles[0].model).to(self.device)  # Init as the first model

        particle_loss_list = torch.zeros(len(self.particles), self.max_iterations)
        particle_accuracy_list = torch.zeros(len(self.particles), self.max_iterations)

        # https://stackoverflow.com/questions/47714643/pytorch-data-loader-multiple-iterations
        # code snippet to repeat the train loader
        # train_generator = iter(self.train_loader)
        # for i in range(1000):
        #    try:
        #        x, y = next(train_generator)
        #    except StopIteration:
        #        train_generator = iter(self.train_loader)
        #        x, y = next(train_generator)

        train_generator = iter(self.train_loader)

        self.model.train()  # Set model to training mode.

        #  Training Loop
        for iteration in range(self.max_iterations):

            # decay coefficient for social and cognitive components from PSO-PINN
            decay = torch.tensor((1 - 1 / self.max_iterations) ** iteration).to(self.device)

            # alternative 1/n for faster and larger decay. This goes to 0 instead of 1/e.
            # decay = torch.tensor(1 / (iteration + 1)).to(self.device)

            # Use training batches for fitness evaluation and for gradient computation.
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

                # calculate gradient with respect to training batch
                # in "https://github.com/caio-davi/PSO-PINN/blob/main/src/swarm/optimizers/fss.py"
                # the fitness function returns gradients i.e. the whole (training) dataset is used
                # We use a training batch to calculate the gradient

                outputs = particle.model(train_inputs)
                loss_fn = torch.nn.CrossEntropyLoss()
                particle.model.zero_grad()
                loss = loss_fn(outputs, train_labels)
                loss.backward()  # gradients are in param_current.grad for param in particle.model.parameters()

                with torch.no_grad():
                    # Update particle velocity and position. The methods with '_' are in place operations.
                    for i, (param_current, g_best, p_best, velocity_current) in enumerate(
                            zip(particle.model.parameters(), global_best_model.parameters(),
                                particle.best_model.parameters(), particle.velocity.parameters())):
                        social_component = decay * self.social_weight * torch.rand(1).to(self.device) * (
                                g_best.data - param_current.data)

                        cognitive_component = decay * self.cognitive_weight * torch.rand(1).to(self.device) * (
                                p_best.data - param_current.data)

                        velocity = velocity_current * self.inertia_weight + \
                                   social_component + \
                                   cognitive_component \
                                   - param_current.grad * self.learning_rate
                        # The velocity is also decaying here unlike in the paper.

                        param_current.add_(velocity)

                # Evaluate particle fitness using the fitness function
                # particle_loss, particle_accuracy = evaluate_position(particle.model, self.valid_loader, self.device)
                particle_loss, particle_accuracy = evaluate_position_single_batch(
                    particle.model, train_inputs, train_labels, self.device)

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
                    global_best_model = copy.deepcopy(particle.model)

                if particle_accuracy > global_best_accuracy:
                    global_best_accuracy = particle_accuracy

                particle.model = particle.model.to("cpu")
                particle.best_model = particle.best_model.to("cpu")
                particle.velocity = particle.velocity.to("cpu")

            if iteration % 20 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Loss: {global_best_loss}")

        if evaluate:
            torch.save(particle_loss_list, "particle_loss_list.pt")
            torch.save(particle_accuracy_list, "particle_accuracy_list.pt")

        # overwrite the initial models parameters
        self.model.load_state_dict(global_best_model.state_dict())

        # give back particles to use as an ensemble
        return [particle.model for particle in self.particles]
