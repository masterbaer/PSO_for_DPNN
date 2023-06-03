import numpy as np
import torch
from sklearn.decomposition import PCA
import copy
from model import *

from torch import nn


# This optimizer is from PSO-PINN. https://arxiv.org/pdf/2202.01943.pdf
# The gradient is used as another velocity component and the social and cognitivce coefficients
# decay with 1/n where n is the number of iterations.

# In contrast to the paper we also let the inertia decay since otherwise the loss explodes (with given inertia).
# At some point, however, this becomes normal SGD with many neural networks in parallel.


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

        # break

    loss_per_batch = valid_loss / number_of_batches
    accuracy = (correct_pred.float() / num_examples).item()
    return loss_per_batch, accuracy


def evaluate_position_single_batch(model, inputs, labels, device):
    #  Compute Loss
    model = model.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    number_of_samples = len(inputs)

    predictions = model(inputs)  # Calculate model output.
    _, predicted = torch.max(predictions, dim=1)  # Determine class with max. probability for each sample.
    num_examples = labels.size(0)  # Update overall number of considered samples.
    correct_pred = (predicted == labels).sum()  # Update overall number of correct predictions.
    loss_total = loss_fn(predictions, labels).item()

    # loss_per_sample = loss / number_of_samples
    accuracy = (correct_pred.float() / num_examples).item()
    return loss_total, accuracy


class Particle:

    def __init__(self, model):
        # maybe only save the parameters?

        # initialize the particle randomly
        self.model = copy.deepcopy(model)
        reset_all_weights(self.model)

        self.velocity = copy.deepcopy(model)
        reset_all_weights(self.velocity)

        self.best_model = copy.deepcopy(self.model)

        self.best_loss = float('inf')  # Change to the best accuracy/fitness?


class PSOWithGradients:
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

    def optimize(self, visualize=False, evaluate=True):

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
            one_over_n = torch.tensor(1 / (iteration + 1)).to(self.device)

            # Use training batches for fitness evaluation and for gradient computation.
            try:
                train_inputs, train_labels = next(train_generator)
            except StopIteration:
                train_generator = iter(self.train_loader)
                train_inputs, train_labels = next(train_generator)
            train_inputs = train_inputs.to(self.device)
            train_labels = train_labels.to(self.device)

            # TODO: maybe give each particle other data for gradient computation?

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
                        social_component = one_over_n * self.social_weight * torch.rand(1).to(self.device) * (
                                g_best.data - param_current.data)

                        cognitive_component = one_over_n * self.cognitive_weight * torch.rand(1).to(self.device) * (
                                p_best.data - param_current.data)

                        velocity = one_over_n * velocity_current * self.inertia_weight + social_component + \
                                   cognitive_component - param_current.grad * self.learning_rate
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

        return global_best_loss, global_best_accuracy
