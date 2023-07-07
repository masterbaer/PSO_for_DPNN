"""
Using only the social component was mentioned by Engelbrecht in his book.
See (https://web2.qatar.cmu.edu/~gdicaro/15382/additional/CompIntelligence-Engelbrecht-ch16.pdf) and search for
"Social-Only" or go to book page 311, pdf-page 23.

Whether there is an improvement or not - we save memory by not saving the cognitive best component. Hence this
alteration is useful regardless.
In contrast, there is also the cognitive-only model where the social component is erased. But that leads to poor
performance as it "tends to locally search in areas where particles are initialized".

He also mentions an inertia decay via Linear Decreasing where "an initially large inertia weight (usually 0.9)
is linearly decreased to a small value (usually 0.4)".
"""

import copy
import time

from model import *

from torch import nn


# This optimizer is from PSO-PINN. https://arxiv.org/pdf/2202.01943.pdf
# The gradient is used as another velocity component and the social and cognitivce coefficients
# decay with 1/n where n is the number of iterations.

# In contrast to the paper we also let the inertia decay since otherwise the loss explodes (with given inertia).
# At some point, however, this becomes normal SGD with many neural networks in parallel.

# To run the different variants, change "decay" in the code.

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
        # initialize the particle randomly
        self.model = copy.deepcopy(model)
        reset_all_weights(self.model)

        # initialize parameters with zeros
        self.velocity = copy.deepcopy(model)
        for param in self.velocity.parameters():
            param.data = torch.zeros_like(param.data)


class PSOWithGradientsOnlySocial:
    def __init__(self,
                 model,
                 num_particles: int,
                 inertia_weight: float,
                 social_weight: float,
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

        self.learning_rate = torch.tensor(learning_rate).to(self.device)
        self.max_iterations = max_iterations
        self.valid_loader = valid_loader
        self.train_loader = train_loader  # for gradient calculation

    def optimize(self, evaluate=True, output1="particle_loss_list.pt", output2="particle_accuracy_list.pt"):

        train_generator = iter(self.train_loader)
        valid_generator = iter(self.valid_loader)

        print("find initial global best model")
        try:
            valid_inputs, valid_labels = next(valid_generator)
        except StopIteration:
            valid_generator = iter(self.valid_loader)
            valid_inputs, valid_labels = next(valid_generator)
        valid_inputs = valid_inputs.to(self.device)
        valid_labels = valid_labels.to(self.device)

        global_best_loss = float("inf")
        global_best_model = copy.deepcopy(self.particles[0].model).to(self.device)

        for particle_index, particle in enumerate(self.particles):
            particle.model.to(self.device)
            particle_loss, particle_accuracy = evaluate_position_single_batch(particle.model, valid_inputs,
                                                                              valid_labels, self.device)
            if particle_loss < global_best_loss:
                global_best_loss = particle_loss
                global_best_model = copy.deepcopy(particle.model).to(self.device)

            particle.model.to("cpu")

        global_best_accuracy = 0.0
        particle_loss_list = torch.zeros(len(self.particles), self.max_iterations)
        particle_accuracy_list = torch.zeros(len(self.particles), self.max_iterations)

        self.model.train()  # Set model to training mode.

        start_time = time.perf_counter()
        #  Training Loop
        for iteration in range(self.max_iterations):

            try:
                valid_inputs, valid_labels = next(valid_generator)
            except StopIteration:
                valid_generator = iter(self.valid_loader)
                valid_inputs, valid_labels = next(valid_generator)
            valid_inputs = valid_inputs.to(self.device)
            valid_labels = valid_labels.to(self.device)

            decay = 1

            # decay = 1 / (1 + torch.log(torch.tensor(iteration + 1))) - \
            #        1 / (1 + torch.log(torch.tensor(self.max_iterations)))

            # Use training batches for fitness evaluation and for gradient computation.

            for particle_index, particle in enumerate(self.particles):
                try:
                    train_inputs, train_labels = next(train_generator)
                except StopIteration:
                    train_generator = iter(self.train_loader)
                    train_inputs, train_labels = next(train_generator)
                train_inputs = train_inputs.to(self.device)
                train_labels = train_labels.to(self.device)

                particle.model = particle.model.to(self.device)
                particle.velocity = particle.velocity.to(self.device)

                outputs = particle.model(train_inputs)
                loss_fn = torch.nn.CrossEntropyLoss()
                particle.model.zero_grad()
                loss = loss_fn(outputs, train_labels)
                loss.backward()

                # Update particle velocity and position. The methods with '_' are in place operations.
                for i, (param_current, g_best, velocity_current) in enumerate(
                        zip(particle.model.parameters(), global_best_model.parameters(),
                            particle.velocity.parameters())):
                    social_component = decay * self.social_weight * torch.rand(1).to(self.device) * (
                            g_best.data - param_current.data)

                    velocity = velocity_current.data * self.inertia_weight + \
                               social_component - param_current.grad * self.learning_rate

                    param_current.data.add_(velocity)
                    velocity_current.data = velocity

                # Evaluate particle fitness using the fitness function
                # particle_loss, particle_accuracy = evaluate_position(particle.model, self.valid_loader, self.device)
                particle_loss, particle_accuracy = evaluate_position_single_batch(
                    particle.model, valid_inputs, valid_labels, self.device)

                if evaluate:
                    particle_loss_list[particle_index, iteration] = particle_loss
                    particle_accuracy_list[particle_index, iteration] = particle_accuracy

                if iteration % 20 == 0:
                    print(f"(particle,loss,accuracy) = "
                          f"{(particle_index + 1, round(particle_loss, 3), round(particle_accuracy, 3))}")

                # Update global best position and fitness
                if particle_loss < global_best_loss:
                    global_best_loss = particle_loss
                    global_best_model = copy.deepcopy(particle.model)

                if particle_accuracy > global_best_accuracy:
                    global_best_accuracy = particle_accuracy

                particle.model = particle.model.to("cpu")
                particle.velocity = particle.velocity.to("cpu")

            if iteration % 20 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Loss: {global_best_loss}")
                local_end_time = time.perf_counter()
                print("time elapsed: ", local_end_time - start_time)
        if evaluate:
            torch.save(particle_loss_list, output1)
            torch.save(particle_accuracy_list, output2)

        # overwrite the initial models parameters
        self.model.load_state_dict(global_best_model.state_dict())

        end_time = time.perf_counter()
        print("total time elapsed: ", end_time - start_time)
        # give back particles to use as an ensemble
        return [particle.model for particle in self.particles]
