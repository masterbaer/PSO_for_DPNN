"""
We hope to achieve better results by adding momentum to the optimizer.
"""

# The velocities [v1, v2, ... , vk] of a particle are stored (which include the average-pull component).
# A momentum of coefficient*(v1+v2+...+vk) is added to the update rule.

import copy
import time

from model import *
from mpi4py import MPI
from helperfunctions import evaluate_position_single_batch, reset_all_weights, evaluate_model


class AveragePullMomentum:
    def __init__(self,
                 model,
                 inertia_weight: float,
                 average_pull_weight: float,
                 max_iterations: int,
                 learning_rate: float,
                 valid_loader,
                 train_loader,
                 device,
                 rank,
                 world_size,
                 comm=MPI.COMM_WORLD,
                 step=10,
                 init_strat="equal",
                 momentum_queue_size=10,
                 momentum_coefficient=0.005,
                 use_average_pull_momentum=True):
        self.use_average_pull_momentum = use_average_pull_momentum
        self.device = device
        self.comm = comm
        self.rank = rank
        self.world_size = world_size

        self.model = model.to(self.device)
        self.average_model = copy.deepcopy(model).to(self.device)

        self.velocity = copy.deepcopy(model).to(self.device)
        for param in self.velocity.parameters():
            param.data = torch.zeros_like(param.data)

        self.momentum_coefficient = torch.tensor(momentum_coefficient).to(self.device)
        # store the cumulative sum of the gradients and velocities in the momentum vector
        # an update sequence of [g1,g2,g3,...,gk,u1] with gi being the gradients and v1 being the velocity
        # would result in momentum = g1+g2...gk+u1. When a new entry occurs,
        # g1 is subtracted and the new entry is added.
        self.momentum = copy.deepcopy(model).to(self.device)
        for param in self.momentum.parameters():
            param.data = torch.zeros_like(param.data)

        self.momentum_queue = []
        for i in range(momentum_queue_size):
            momentum_local = copy.deepcopy(self.momentum).to(self.device)
            for param in momentum_local.parameters():
                param.data = torch.zeros_like(param.data)  # start with zeros
            self.momentum_queue.append(momentum_local)

        self.inertia_weight = torch.tensor(inertia_weight).to(self.device)
        self.average_pull_weight = torch.tensor(average_pull_weight).to(self.device)

        self.learning_rate = torch.tensor(learning_rate).to(self.device)
        self.step = step
        self.max_iterations = max_iterations
        self.valid_loader = valid_loader  # for evaluation of fitness
        self.train_loader = train_loader  # for gradient calculation

        if init_strat == "random":
            reset_all_weights(self.model)

        if init_strat == "equal":
            state_dict = self.comm.bcast(self.model.state_dict(), root=0)
            self.model.load_state_dict(state_dict)

    def optimize(self, evaluate=True, output1="global_loss_list.pt", output2="global_accuracy_list.pt"):

        train_generator = iter(self.train_loader)
        valid_loss_list = []
        valid_accuracy_list = []

        start_time = time.perf_counter()
        for iteration in range(self.max_iterations):

            if iteration % self.step == 0 and iteration != 0:
                # synchronization via average pull
                # see https://discuss.pytorch.org/t/average-each-weight-of-two-models/77008 for averaging two
                # models.

                state_dict = self.model.state_dict()

                # convert state_dict values to tensors
                for key in state_dict:
                    state_dict[key] = torch.tensor(state_dict[key])

                # create average model with zeros
                average_state_dict = {key: torch.zeros_like(state_dict[key]) for key in state_dict}

                for key in average_state_dict:
                    value = state_dict[key]  # value to average
                    self.comm.Allreduce(value, average_state_dict[key], op=MPI.SUM)
                    average_state_dict[key] /= self.world_size

                self.average_model.load_state_dict(average_state_dict)

                # perform average pull with momentum

                # update the velocity with the pull towards the average
                for i, (param, param_average, velocity) in enumerate(
                        zip(self.model.parameters(), self.average_model.parameters(), self.velocity.parameters())):
                    average_pull = self.average_pull_weight * (param_average.data - param.data)
                    velocity.data = velocity.data * self.inertia_weight + average_pull

                # update the particle position by adding the velocity
                for param, velocity in zip(self.model.parameters(), self.velocity.parameters()):
                    param.data.add_(velocity)


                # update the momentum by adding the velocity and subtracting the old value and updating the queue
                # pop and subtract the old values
                if self.use_average_pull_momentum:
                    old_net = self.momentum_queue.pop(0)
                    for param_momentum, param_old_value in zip(self.momentum.parameters(), old_net.parameters()):
                        param_momentum.data.sub_(param_old_value.data)

                    # update the old net and append it to the queue again
                    for param_old, velocity_current in zip(old_net.parameters(), self.velocity.parameters()):
                        param_old.data.copy_(velocity_current)
                    self.momentum_queue.append(old_net)

                    # add the new values to the current momentum
                    for param_momentum, param_new_value in zip(self.momentum.parameters(), old_net.parameters()):
                        param_momentum.data.add_(param_new_value.data)

                # update the particle position by adding the momentum
                for param, momentum in zip(self.model.parameters(), self.momentum.parameters()):
                    param.data.add_(momentum * self.momentum_coefficient)

            else:
                # local SGD update
                try:
                    train_inputs, train_labels = next(train_generator)
                except StopIteration:
                    train_generator = iter(self.train_loader)
                    train_inputs, train_labels = next(train_generator)
                train_inputs = train_inputs.to(self.device)
                train_labels = train_labels.to(self.device)

                outputs = self.model(train_inputs)
                loss_fn = torch.nn.CrossEntropyLoss()
                self.model.zero_grad()
                loss = loss_fn(outputs, train_labels)
                loss.backward()

                # update model parameters
                for param_current in self.model.parameters():
                    param_current.data.sub_(param_current.grad * self.learning_rate)

                # update momentum by adding the new gradients and subtracting the oldest values and updating the queue
                # subtract the old values and pop them from the queue
                old_net = self.momentum_queue.pop(0)
                for param_momentum, param_old_value in zip(self.momentum.parameters(), old_net.parameters()):
                    param_momentum.data.sub_(param_old_value.data)

                # update the old net and append it to the queue again
                for param_old, param_current in zip(old_net.parameters(), self.model.parameters()):
                    param_old.data.copy_(param_current.grad)
                self.momentum_queue.append(old_net)

                # add the new values to the current momentum
                for param_momentum, param_new_value in zip(self.momentum.parameters(), old_net.parameters()):
                    param_momentum.data.add_(param_new_value.data)

            if iteration % 20 == 0:
                # validation accuracy on first particle
                end_time_iteration = time.perf_counter()
                particle_loss, particle_accuracy = evaluate_model(self.model, self.valid_loader, self.device)
                valid_loss_list.append(particle_loss)
                valid_accuracy_list.append(particle_accuracy)

                if self.rank == 0:
                    print(f"iteration {iteration + 1}, accuracy: {particle_accuracy}")
                    print(f"time passed: {end_time_iteration - start_time}")

        # After training

        if evaluate and self.rank == 0:
            torch.save(valid_loss_list, output1)
            torch.save(valid_accuracy_list, output2)

        if self.rank == 0:
            end_time = time.perf_counter()
            print("total time elapsed for training: ", end_time - start_time)
