

import copy
import time

from model import *
from mpi4py import MPI
from helperfunctions import evaluate_position_single_batch, reset_all_weights, evaluate_model

"""
We delete the cognitive component again, replace the global best particle with the "current" best particle
and increase the social weight so that all particles are both
a) pulled towards the average
b) pulled towards the current best particle.
"""

class PSOAveragePullv2:
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
                 init_strat="equal",
                 social_weight=0.9):
        self.device = device
        self.comm = comm
        self.rank = rank
        self.world_size = world_size

        self.model = model

        self.velocity = copy.deepcopy(model)
        for param in self.velocity.parameters():
            param.data = torch.zeros_like(param.data)

        self.inertia_weight = torch.tensor(inertia_weight)
        self.average_pull_weight = torch.tensor(average_pull_weight)

        self.learning_rate = torch.tensor(learning_rate)
        self.max_iterations = max_iterations
        self.social_weight = social_weight
        self.valid_loader = valid_loader  # for evaluation of fitness
        self.train_loader = train_loader  # for gradient calculation

        if init_strat == "random":
            reset_all_weights(self.model)

        if init_strat == "equal":
            state_dict = self.comm.bcast(self.model.state_dict(), root=0)
            self.model.load_state_dict(state_dict)

        self.current_best = copy.deepcopy(self.model)
        for param_cbest, param in zip(self.current_best.parameters(), self.model.parameters()):
            param_cbest.data.copy_(param.data)

        self.current_best_loss = float("inf")

    def optimize(self, evaluate=True, output1="global_loss_list.pt", output2="global_accuracy_list.pt"):

        train_generator = iter(self.train_loader)
        valid_loss_list = []
        valid_accuracy_list = []

        start_time = time.perf_counter()

        for iteration in range(self.max_iterations):

            decay = torch.tensor((1 - 1 / self.max_iterations) ** iteration) # linear decay

            # calculate the gradient and update the particle
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
            self.current_best_loss = loss.item()

            for param in self.model.parameters():
                param.data.sub_(param.grad * self.learning_rate)

            # identify current best

            min_value, min_rank = self.comm.allreduce((self.current_best_loss, self.rank), op=MPI.MINLOC)
            self.current_best_loss = min_value
            current_best_model_dict = self.comm.bcast(self.model.state_dict(), root=min_rank)
            self.current_best.load_state_dict(current_best_model_dict)

            # calculate the average particle
            state_dict = self.model.state_dict()
            average_model = copy.deepcopy(self.model).to(self.device)
            for param in average_model.parameters():
                param.data = torch.zeros_like(param.data)
            average_state_dict = average_model.state_dict()

            for key in average_state_dict:
                value = state_dict[key]  # value to average
                summed_value = self.comm.allreduce(value, op=MPI.SUM)
                average_state_dict[key] = summed_value / self.world_size
                average_model.load_state_dict(average_state_dict)


            for i, (param_current, param_average, velocity_current, current_best) in enumerate(
                    zip(self.model.parameters(), average_model.parameters(), self.velocity.parameters(),
                        self.current_best.parameters())):

                average_pull = self.average_pull_weight * (param_average.data - param_current.data)

                current_social_component = decay * self.social_weight * torch.rand(1) * (
                        current_best.data - param_current.data)

                velocity = velocity_current.data * self.inertia_weight + average_pull\
                           + current_social_component

                param_current.data.add_(velocity)
                velocity_current.data.copy_(velocity)

            if iteration % 20 == 0:
                # validation accuracy on first particle
                end_time_iteration = time.perf_counter()
                particle_loss, particle_accuracy = evaluate_model(self.current_best, self.valid_loader, self.device)
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