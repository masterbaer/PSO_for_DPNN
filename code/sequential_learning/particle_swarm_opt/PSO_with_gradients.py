import copy
import time

from model import *
from helperfunctions import reset_all_weights, evaluate_position_single_batch


# This optimizer is from PSO-PINN. https://arxiv.org/pdf/2202.01943.pdf
# The gradient is used as another velocity component and the social and cognitive coefficients
# decay with 1/n where n is the number of iterations.

# In contrast to the paper we also let the inertia decay since otherwise the loss explodes (with given inertia).
# At some point, however, this becomes normal SGD with many neural networks in parallel.

# To run the different variants, change "decay" in the code.


# The particle evaluation is similar to the validation process.
# see "Per-Epoch Activity" https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

class Particle:

    def __init__(self, model):
        # initialize the particle randomly
        self.model = copy.deepcopy(model)
        reset_all_weights(self.model)

        # initialize parameters with zeros
        self.velocity = copy.deepcopy(model)
        for param in self.velocity.parameters():
            param.data = torch.zeros_like(param.data)

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

    def optimize(self, evaluate=True):

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

        global_best_accuracy = 0.0
        particle_loss_list = torch.zeros(len(self.particles), self.max_iterations)
        particle_accuracy_list = torch.zeros(len(self.particles), self.max_iterations)

        self.model.train()  # Set model to training mode.

        start_time = time.perf_counter()
        #  Training Loop
        for iteration in range(self.max_iterations):

            # decay coefficient for social and cognitive components from PSO-PINN
            # decay = torch.tensor((1 - 1 / self.max_iterations) ** iteration).to(self.device)

            # alternative 1/n for faster and larger decay. This goes to 0 instead of 1/e.
            # decay = torch.tensor(1 / (iteration + 1)).to(self.device)

            # decay with 1/ (1 + log(n+1)) for n = 1 ...
            #decay = 1 / (1 + torch.log(torch.tensor(iteration + 1)))

            # subtract the last value so the decay goes down to zero
            decay = 1 / (1 + torch.log(torch.tensor(iteration + 1))) - \
                    1 / (1 + torch.log(torch.tensor(self.max_iterations)))

            # velocity only contains inertia and the gradient
            #decay = 0

            # Use training batches for fitness evaluation and for gradient computation.

            try:
                valid_inputs, valid_labels = next(valid_generator)
            except StopIteration:
                valid_generator = iter(self.valid_loader)
                valid_inputs, valid_labels = next(valid_generator)
            valid_inputs = valid_inputs.to(self.device)
            valid_labels = valid_labels.to(self.device)

            for particle_index, particle in enumerate(self.particles):

                try:
                    train_inputs, train_labels = next(train_generator)
                except StopIteration:
                    train_generator = iter(self.train_loader)
                    train_inputs, train_labels = next(train_generator)
                train_inputs = train_inputs.to(self.device)
                train_labels = train_labels.to(self.device)

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
                    particle.model, valid_inputs, valid_labels, self.device)

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
                local_end_time = time.perf_counter()
                print("time elapsed: ", local_end_time - start_time)

        if evaluate:
            torch.save(particle_loss_list, "particle_loss_list.pt")
            torch.save(particle_accuracy_list, "particle_accuracy_list.pt")

        # overwrite the initial models parameters
        self.model.load_state_dict(global_best_model.state_dict())

        end_time = time.perf_counter()
        print("total time elapsed: ", end_time-start_time)
        # give back particles to use as an ensemble
        return [particle.model for particle in self.particles]
