import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA, PCA


# Adapted from the torch-pso implementation
# See https://github.com/qthequartermasterman/torch_pso/blob/master/torch_pso/optim/ParticleSwarmOptimizer.py

def extract_weights(model):
    weights = []
    for param in model.parameters():
        weights.append(param.data.view(-1))
    return torch.cat(weights)


def set_weights(model, flattened_weights):
    start = 0
    for param in model.parameters():
        end = start + param.numel()
        param.data.copy_(flattened_weights[start:end].reshape(param.shape))
        start = end


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

    loss_per_batch = valid_loss / number_of_batches
    accuracy = correct_pred.float() / num_examples
    return loss_per_batch, accuracy


class Particle:

    def __init__(self, num_weights, min_param_value, max_param_value):
        # maybe only save the parameters?
        self.position = (max_param_value - min_param_value) * torch.rand(num_weights) + min_param_value
        self.velocity = (max_param_value - min_param_value) * torch.rand(num_weights) + min_param_value
        self.best_position = self.position.detach().clone()
        self.best_loss = float('inf')  # Change to the best accuracy/fitness?


class PSO:
    def __init__(self,
                 model,
                 num_particles: int,
                 inertia_weight: float,
                 social_weight: float,
                 cognitive_weight: float,
                 min_param_value: float,
                 max_param_value: float,
                 max_iterations: int,
                 train_loader,
                 device):
        self.model = model
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.social_weight = social_weight
        self.cognitive_weight = cognitive_weight
        self.min_param_value = min_param_value
        self.max_param_value = max_param_value
        self.max_iterations = max_iterations
        self.train_loader = train_loader
        self.device = device

    def optimize(self):
        num_weights = sum(p.numel() for p in self.model.parameters())
        particles = [Particle(num_weights, self.min_param_value, self.max_param_value) for _ in
                     range(self.num_particles)]
        global_best_loss = float('inf')
        global_best_accuracy = 0.0
        global_best_position = particles[0].best_position  # init as first particle position

        particle_loss_list = torch.zeros(len(particles), self.max_iterations)
        particle_accuracy_list = torch.zeros(len(particles), self.max_iterations)

        # ipca = IncrementalPCA(n_components=2, batch_size=len(particles))
        pca = PCA(n_components=2)
        particles_transformed = np.zeros((self.max_iterations, len(particles), 2))

        for iteration in range(self.max_iterations):
            for particle_index, particle in enumerate(particles):
                # Update particle velocity and position
                particle.velocity = (self.inertia_weight * particle.velocity +
                                     self.cognitive_weight * torch.rand(1) *
                                     (particle.best_position - particle.position) +
                                     self.social_weight * torch.rand(1) *
                                     (global_best_position - particle.position))
                particle.position += particle.velocity

                # Update neural network weights using the particle's position
                set_weights(self.model, particle.position)

                # Evaluate particle fitness using the fitness function
                particle_loss, particle_accuracy = evaluate_position(self.model, self.train_loader, self.device)
                particle_loss_list[particle_index, iteration] = particle_loss
                particle_accuracy_list[particle_index, iteration] = particle_accuracy

                print(
                    f"(particle,loss,accuracy) = {(particle_index + 1, round(particle_loss, 3), round(particle_accuracy.item(), 3))}")

                # Update particle's best position and fitness
                if particle_loss < particle.best_loss:
                    particle.best_loss = particle_loss
                    particle.best_position = particle.position.clone()

                # Update global best position and fitness
                if particle_loss < global_best_loss:
                    global_best_loss = particle_loss
                    global_best_position = particle.position.clone()

                if particle_accuracy > global_best_accuracy:
                    global_best_accuracy = particle_accuracy

            particles_np = np.zeros((len(particles), num_weights))
            for particle_index, particle in enumerate(particles):
                particles_np[particle_index] = particle.position.numpy()
            print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Loss: {global_best_loss}")

            # print("using ipca to visualise data")
            print("using pca to visualise data")
            if iteration == 0:
                # ipca.partial_fit(particles_np)
                pca.fit(particles_np)

            print("transforming data", )
            # particles_transformed[iteration] = ipca.transform(particles_np)
            particles_transformed[iteration] = pca.transform(particles_np)

        torch.save(particle_loss_list, "particle_loss_list.pt")
        torch.save(particle_accuracy_list, "particle_accuracy_list.pt")
        torch.save(particles_transformed, "pca_weights.pt")
