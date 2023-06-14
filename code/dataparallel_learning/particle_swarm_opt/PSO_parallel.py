import copy
from model import *
from mpi4py import MPI
from torch import nn


# TODO parallelize

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
    #  Compute Loss locally
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
    #  Compute Loss locally for a single batch
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

        self.best_model = copy.deepcopy(self.model)  # personal best model

        self.best_loss = float('inf')


class PSO_parallel:
    def __init__(self,
                 model,
                 particles_per_rank: int,
                 inertia_weight: float,
                 social_weight: float,
                 cognitive_weight: float,
                 max_iterations: int,
                 learning_rate: float,
                 valid_loader,
                 train_loader,
                 device,
                 rank,
                 world_size,
                 comm=MPI.COMM_WORLD):
        self.device = device
        self.comm = comm
        self.rank = rank
        self.world_size = world_size

        self.model = model

        self.particles_per_rank = particles_per_rank

        self.particles = []
        for i in range(self.particles_per_rank):
            self.particles.append(Particle(self.model))  # each rank contains multiple particles

        self.inertia_weight = torch.tensor(inertia_weight).to(self.device)
        self.social_weight = torch.tensor(social_weight).to(self.device)
        self.cognitive_weight = torch.tensor(cognitive_weight).to(self.device)

        self.learning_rate = torch.tensor(learning_rate).to(self.device)
        self.max_iterations = max_iterations
        self.valid_loader = valid_loader  # for evaluation of fitness
        self.train_loader = train_loader  # for gradient calculation

    def optimize(self, evaluate=True):

        global_best_loss = float('inf')
        global_best_accuracy = 0.0

        global_best_model = copy.deepcopy(self.particles[0].model).to(self.device)  # Init as the first model

        # the losses and accuracies
        global_loss_list = None
        global_accuracy_list = None
        if self.rank == 0:
            global_loss_list = torch.zeros(self.max_iterations)
            global_accuracy_list = torch.zeros(self.max_iterations)

        train_generator = iter(self.train_loader)

        self.model.train()  # Set model to training mode.

        #  Training Loop
        for iteration in range(self.max_iterations):

            decay = 1 / (1 + torch.log(torch.tensor(iteration + 1))) - \
                    1 / (1 + torch.log(torch.tensor(self.max_iterations)))

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

                outputs = particle.model(train_inputs)
                loss_fn = torch.nn.CrossEntropyLoss()
                particle.model.zero_grad()
                loss = loss_fn(outputs, train_labels)
                loss.backward()

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

                        param_current.add_(velocity)

                # Evaluate particle fitness using the fitness function
                # particle_loss, particle_accuracy = evaluate_position(particle.model, self.valid_loader, self.device)
                particle_loss, particle_accuracy = evaluate_position_single_batch(
                    particle.model, train_inputs, train_labels, self.device)

                # Update particle's best position and fitness
                if particle_loss < particle.best_loss:
                    particle.best_loss = particle_loss
                    particle.best_model = copy.deepcopy(particle.model)

                # We need to broadcast the information that a better model was found.
                found_better_gbest = False

                if particle_loss < global_best_loss:
                    global_best_loss = particle_loss
                    found_better_gbest = True

                # check if the best current rank has in improvement
                min_value, min_rank = self.comm.allreduce(global_best_loss, op=MPI.MINLOC)
                found_better_gbest = self.comm.bcast(found_better_gbest, root=min_rank)

                if found_better_gbest:
                    global_best_loss = self.comm.bcast(min_value, root=min_rank)
                    global_best_accuracy = self.comm.bcast(particle_accuracy, root=min_rank)
                    # everyone shall have the updated global best model
                    global_best_model_dict = self.comm.bcast(particle.model.state_dict(), root=min_rank)
                    global_best_model.load_state_dict(global_best_model_dict)

                    # global_best_model = copy.deepcopy(particle.model)

                particle.model = particle.model.to("cpu")
                particle.best_model = particle.best_model.to("cpu")
                particle.velocity = particle.velocity.to("cpu")

            if iteration % 20 == 0 and self.rank == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Loss: {global_best_loss}")

            # save the best loss and accuracy of all combined particles in each iteration
            if evaluate and self.rank == 0:
                global_loss_list[iteration] = global_best_loss
                global_accuracy_list[iteration] = global_best_accuracy

        if evaluate and self.rank == 0:
            torch.save(global_loss_list, "global_loss_list.pt")
            torch.save(global_accuracy_list, "global_accuracy_list.pt")

        # overwrite the initial models parameters on root
        if self.rank == 0:
            self.model.load_state_dict(global_best_model.state_dict())

        # give back particles to use as an ensemble
        return [particle.model for particle in self.particles]
