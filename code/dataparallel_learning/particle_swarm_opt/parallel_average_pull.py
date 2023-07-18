"""
We try this synchronization method as we see no acceleration in using PSO in parallel as compared to using
simple SGD for one particle only.
The reason could be that PSO updates do not make full use of all the data that the particles train on.
When the social component pulls the particles towards the best one, the old particle positions can be "forgotten".
Also, when the global best changes, this happens as well for the previous best particle.
We want to make use of all the available data.

Another issue might be that the particles are initialized differently. The space between the particles, where the
particles are pulled towards, might not be useful.

For these reasons we try a new approach:
First, initialize all particles similarly (try out random init as well).
Train all particles locally for some iterations on different data.
Then we do not create a pull towards the global best particle, but instead towards the average particle.
This is somewhat similar to Synchronous SGD where the model parameters are averaged.
The difference is that the particles are only pulled towards the average and are still different from it.
"""

import copy
import time

from model import *
from mpi4py import MPI
from helperfunctions import evaluate_position_single_batch, reset_all_weights, evaluate_model


class AveragePull:
    def __init__(self,
                 model,
                 inertia_weight: float,
                 social_weight: float,
                 max_iterations: int,
                 learning_rate: float,
                 valid_loader,
                 train_loader,
                 device,
                 rank,
                 world_size,
                 comm=MPI.COMM_WORLD,
                 step=10,
                 init_strat="equal"):
        self.device = device
        self.comm = comm
        self.rank = rank
        self.world_size = world_size

        self.model = model.to(self.device)

        self.velocity = copy.deepcopy(model).to(self.device)
        for param in self.velocity.parameters():
            param.data = torch.zeros_like(param.data)

        self.inertia_weight = torch.tensor(inertia_weight).to(self.device)
        self.social_weight = torch.tensor(social_weight).to(self.device)

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

        def tensor_add(a, b, datatype):
            a.add_(b)
            return a

        tensor_add_Op = MPI.Op.Create(tensor_add, commute=True)

        for iteration in range(self.max_iterations):

            if iteration % self.step == 0 and iteration != 0:
                # synchronization via average pull
                # see https://discuss.pytorch.org/t/average-each-weight-of-two-models/77008 for averaging two
                # models.

                state_dict = self.model.state_dict()

                average_model = copy.deepcopy(self.model).to(self.device)
                for param in average_model.parameters():
                    param.data = torch.zeros_like(param.data)

                average_state_dict = average_model.state_dict()

                # create a tensor addition operation for mpi

                # this works
                # for param_current, param_average in zip((self.model.parameters()), average_model.parameters()):
                #    value = param_current.data
                #    summed_value = torch.zeros_like(param_current.data)
                #    summed_value = self.comm.allreduce(value, op=tensor_add_Op)
                #    averaged_value = summed_value / self.world_size
                #    param_average.data.copy_(averaged_value)

                # checking this for sanity again
                for key in average_state_dict:
                    value = state_dict[key]  # value to average
                    summed_value = self.comm.allreduce(value, op=MPI.SUM)
                    average_state_dict[key] = summed_value / self.world_size
                    average_model.load_state_dict(average_state_dict)

                # give the first values as a sanity check
                if iteration <= 15:
                    for param in self.model.parameters():
                        print(f"rank {self.rank} and value {param.data[0][0]}")
                        break
                    for param in average_model.parameters():
                        print(f"averaged value: {param.data[0][0]}")
                        break

                # perform average pull

                for i, (param_current, param_average, velocity_current) in enumerate(
                        zip(self.model.parameters(), average_model.parameters(), self.velocity.parameters())):

                    if iteration <= 15 and self.rank == 0:
                        print("param_current: ", param_current.data[0][0])
                        print("param_average: ", param_average.data[0][0])
                        print("social_weight: ", self.social_weight)
                        print("velocity_current: ", velocity_current.data[0][0])
                        print("inertia weight: ", self.inertia_weight)

                    average_pull = self.social_weight * (param_average.data - param_current.data)

                    velocity = velocity_current.data * self.inertia_weight + average_pull
                    param_current.data.add_(velocity)
                    velocity_current.data.copy_(velocity)

                    if iteration <= 15 and self.rank == 0:
                        print("after update: ")
                        print("velocity-term: ", velocity[0][0])
                        print("new velocity: ", velocity_current.data[0][0])
                        print("new param: ", param_current.data[0][0])

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

                for param_current in self.model.parameters():
                    param_current.data.sub_(param_current.grad * self.learning_rate)

            if iteration % 5 == 0:
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
