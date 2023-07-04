"""
This particle swarm optimizer is called PSO-PS (https://arxiv.org/pdf/2009.03816.pdf).
Here, PSO is used for synchronization between particles. The particles themselves use SGD locally.
In contrast to PSO-PINN, PSO-PS does not mix PSO and gradients but instead uses them sequentially.
Also, the influence of SGD is much higher in PSO-PS.

The authors used only one particle per rank in their experiments.
We use multiple particles as we want to analyze the scalability of this approach.
Hence we have the overhead of switching out particles.
"""

import copy
import time

from model import *
from mpi4py import MPI
from helperfunctions import evaluate_position_single_batch, reset_all_weights


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


class PSO_PS:
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
                 comm=MPI.COMM_WORLD,
                 step=10):
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
        self.step = step
        self.max_iterations = max_iterations
        self.valid_loader = valid_loader  # for evaluation of fitness
        self.train_loader = train_loader  # for gradient calculation

    def optimize(self, evaluate=True, output1="global_loss_list.pt", output2="global_accuracy_list.pt"):

        train_generator = iter(self.train_loader)
        valid_generator = iter(self.valid_loader)

        # find initial global best model
        try:
            valid_inputs, valid_labels = next(valid_generator)
        except StopIteration:
            valid_generator = iter(self.valid_loader)
            valid_inputs, valid_labels = next(valid_generator)
        valid_inputs = valid_inputs.to(self.device)
        valid_labels = valid_labels.to(self.device)

        for particle_index, particle in enumerate(self.particles):
            particle.model.to(self.device)
            particle_loss, particle_accuracy = evaluate_position_single_batch(particle.model, valid_inputs,
                                                                              valid_labels, self.device)
            particle.best_loss = particle_loss
            particle.model.to("cpu")

        global_best_loss = self.particles[0].best_loss
        global_best_model = copy.deepcopy(self.particles[0].model).to(self.device)
        for particle_index, particle in enumerate(self.particles):
            if particle.best_loss < global_best_loss:
                global_best_loss = particle.best_loss
                global_best_model = copy.deepcopy(particle.model).to(self.device)

        # synchronize global best model across all ranks
        min_value, min_rank = self.comm.allreduce((global_best_loss, self.rank), op=MPI.MINLOC)
        global_best_loss = self.comm.bcast(min_value, root=min_rank)
        global_best_model_dict = self.comm.bcast(global_best_model.state_dict(), root=min_rank)
        global_best_model.load_state_dict(global_best_model_dict)

        global_best_accuracy = 0.0

        # the losses and accuracies
        global_loss_list = None
        global_accuracy_list = None
        if self.rank == 0:
            global_loss_list = torch.zeros(self.max_iterations)
            global_accuracy_list = torch.zeros(self.max_iterations)

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

            decay = torch.tensor((1 - 1 / self.max_iterations) ** iteration).to(self.device)

            # PSO-synchronization every "step" steps
            if iteration % self.step == 0:

                # evaluate first, use PSO velocity update later as in the paper
                for particle_index, particle in enumerate(self.particles):

                    particle.model = particle.model.to(self.device)
                    particle.best_model = particle.best_model.to(self.device)

                    particle_loss, particle_accuracy = evaluate_position_single_batch(
                        particle.model, valid_inputs, valid_labels, self.device)

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
                    # using MPI_MINLOC requires a 2-tuple. See https://groups.google.com/g/mpi4py/c/XnOEknuSzbs.
                    min_value, min_rank = self.comm.allreduce((global_best_loss, self.rank), op=MPI.MINLOC)
                    found_better_gbest = self.comm.bcast(found_better_gbest, root=min_rank)

                    if found_better_gbest:
                        global_best_loss = self.comm.bcast(min_value, root=min_rank)
                        global_best_accuracy = self.comm.bcast(particle_accuracy, root=min_rank)
                        # everyone shall have the updated global best model
                        global_best_model_dict = self.comm.bcast(particle.model.state_dict(), root=min_rank)
                        global_best_model.load_state_dict(global_best_model_dict)

                    particle.model = particle.model.to("cpu")
                    particle.best_model = particle.best_model.to("cpu")

                # do the PSO-update
                for particle_index, particle in enumerate(self.particles):

                    particle.model = particle.model.to(self.device)
                    particle.best_model = particle.best_model.to(self.device)
                    particle.velocity = particle.velocity.to(self.device)

                    with torch.no_grad():
                        # Update particle velocity and position. The methods with '_' are in place operations.
                        for i, (param_current, g_best, p_best, velocity_current) in enumerate(
                                zip(particle.model.parameters(), global_best_model.parameters(),
                                    particle.best_model.parameters(), particle.velocity.parameters())):
                            social_component = decay * self.social_weight * torch.rand(1).to(self.device) * (
                                    g_best.data - param_current.data)

                            cognitive_component = decay * self.cognitive_weight * torch.rand(1).to(self.device) * (
                                    p_best.data - param_current.data)

                            # no gradient here
                            velocity = velocity_current * self.inertia_weight + social_component + cognitive_component
                            param_current.add_(velocity)

                    particle.model = particle.model.to("cpu")
                    particle.best_model = particle.best_model.to("cpu")
                    particle.velocity = particle.velocity.to("cpu")

            else:
                # perform SGD-updates on all particles
                for particle_index, particle in enumerate(self.particles):
                    particle.model = particle.model.to(self.device)

                    try:
                        train_inputs, train_labels = next(train_generator)
                    except StopIteration:
                        train_generator = iter(self.train_loader)
                        train_inputs, train_labels = next(train_generator)
                    train_inputs = train_inputs.to(self.device)
                    train_labels = train_labels.to(self.device)

                    outputs = particle.model(train_inputs)
                    loss_fn = torch.nn.CrossEntropyLoss()
                    particle.model.zero_grad()
                    loss = loss_fn(outputs, train_labels)
                    loss.backward()

                    for param_current in particle.model.parameters():
                        param_current.data.sub_(param_current.grad * self.learning_rate)

                    particle.model = particle.model.to("cpu")

            if iteration % 20 == 0 and self.rank == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Loss: {global_best_loss}, "
                      f"accuracy: {global_best_accuracy}")

            # save the best loss and accuracy of all combined particles in each iteration
            if evaluate and self.rank == 0:
                global_loss_list[iteration] = global_best_loss
                global_accuracy_list[iteration] = global_best_accuracy

            end_time_local = time.perf_counter()
            if iteration % 20 == 0 and self.rank == 0:
                print(f"time elapsed after {iteration + 1} iterations: ", end_time_local - start_time)

        # After training

        if evaluate and self.rank == 0:
            torch.save(global_loss_list, output1)
            torch.save(global_accuracy_list, output2)

        if self.rank == 0:
            end_time = time.perf_counter()
            print("total time elapsed for training: ", end_time - start_time)

        # overwrite the initial models parameters on root
        if self.rank == 0:
            self.model.load_state_dict(global_best_model.state_dict())

        # give back particles to use as an ensemble
        return [particle.model for particle in self.particles]
