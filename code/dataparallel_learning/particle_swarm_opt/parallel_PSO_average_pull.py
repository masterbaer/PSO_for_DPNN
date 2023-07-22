"""
In this PSO variant we try to combine the PSO-PINN approach with another average-pull component.
In each iteration, we use the average-pull component, the inertia and a local gradient.

"""
# TODO make it a PSO-variant by adding the personal best and global best!

import copy
import time

from model import *
from mpi4py import MPI
from helperfunctions import evaluate_position_single_batch, reset_all_weights, evaluate_model


class PSOAveragePull:
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
                 social_weight=0.0,
                 cognitive_weight=0.0):
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
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.valid_loader = valid_loader  # for evaluation of fitness
        self.train_loader = train_loader  # for gradient calculation

        if init_strat == "random":
            reset_all_weights(self.model)

        if init_strat == "equal":
            state_dict = self.comm.bcast(self.model.state_dict(), root=0)
            self.model.load_state_dict(state_dict)

        self.pbest = copy.deepcopy(self.model)
        for param_pbest, param in zip(self.pbest.parameters(), self.model.parameters()):
            param_pbest.data.copy_(param.data)

        self.gbest = copy.deepcopy(self.model)
        for param_gbest, param in zip(self.gbest.parameters(), self.model.parameters()):
            param_gbest.data.copy_(param.data)

        self.personal_best_loss = float("inf")
        self.global_best_loss = float("inf")

    def optimize(self, evaluate=True, output1="global_loss_list.pt", output2="global_accuracy_list.pt"):

        train_generator = iter(self.train_loader)
        valid_loss_list = []
        valid_accuracy_list = []

        start_time = time.perf_counter()

        for iteration in range(self.max_iterations):

            decay = torch.tensor((1 - 1 / self.max_iterations) ** iteration) # linear decay

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

            # calculate the gradient
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
            loss_value = loss.item()

            # update pbest
            if loss_value < self.personal_best_loss:
                self.personal_best_loss = loss_value
                state_dict = self.model.state_dict()
                self.pbest.load_state_dict(state_dict)

            # update gbest
            found_better_gbest = False

            if loss_value < self.global_best_loss:
                self.global_best_loss = loss_value
                found_better_gbest = True

            min_value, min_rank = self.comm.allreduce((self.global_best_loss, self.rank), op=MPI.MINLOC)
            found_better_gbest = self.comm.bcast(found_better_gbest, root=min_rank)

            if found_better_gbest:
                self.global_best_loss = self.comm.bcast(min_value, root=min_rank)
                #global_best_accuracy = self.comm.bcast(particle_accuracy, root=min_rank)
                global_best_model_dict = self.comm.bcast(self.model.state_dict(), root=min_rank)
                self.gbest.load_state_dict(global_best_model_dict)

            for i, (param_current, param_average, velocity_current, p_best, g_best) in enumerate(
                    zip(self.model.parameters(), average_model.parameters(), self.velocity.parameters(),
                        self.pbest.parameters(), self.gbest.parameters())):

                average_pull = self.average_pull_weight * (param_average.data - param_current.data)
                gradient_component = param_current.grad * self.learning_rate

                social_component = decay * self.social_weight * torch.rand(1) * (
                        g_best.data - param_current.data)

                cognitive_component = decay * self.cognitive_weight * torch.rand(1) * (
                        p_best.data - param_current.data)

                velocity = velocity_current.data * self.inertia_weight + average_pull - gradient_component \
                           + social_component + cognitive_component

                param_current.data.add_(velocity)
                velocity_current.data.copy_(velocity)

            if iteration % 20 == 0:
                # validation accuracy on first particle
                end_time_iteration = time.perf_counter()
                particle_loss, particle_accuracy = evaluate_model(self.gbest, self.valid_loader, self.device)
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
