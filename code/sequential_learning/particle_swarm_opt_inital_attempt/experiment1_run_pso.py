import torch
import torchvision

from sequential_learning.particle_swarm_opt_inital_attempt.PSO import PSO
from sequential_learning.particle_swarm_opt_inital_attempt.dataloader import get_dataloaders_cifar10


def evaluate_model(model, test_data_loader):
    #  Compute Loss
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    number_of_batches = len(test_data_loader)
    correct_pred, num_examples = 0, 0

    for i, (vinputs, vlabels) in enumerate(test_data_loader):  # Loop over batches in data.
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        predictions = model(vinputs)  # Calculate model output.
        _, predicted = torch.max(predictions, dim=1)  # Determine class with max. probability for each sample.
        num_examples += vlabels.size(0)  # Update overall number of considered samples.
        correct_pred += (predicted == vlabels).sum()  # Update overall number of correct predictions.
        loss = loss_fn(predictions, vlabels)
        test_loss = test_loss + loss.item()

    loss_per_batch = test_loss / number_of_batches
    accuracy = (correct_pred.float() / num_examples).item()
    return loss_per_batch, accuracy


if __name__ == '__main__':

    b = 256  # Set batch size.

    # Get device used for training, e.g., check via torch.cuda.is_available().
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    print(f'Using {device} device.')

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.CenterCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # GET PYTORCH DATALOADERS FOR TRAINING, TESTING, AND VALIDATION DATASET.
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        root="../../data",
        validation_fraction=0.1,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        num_workers=0
    )

    # Check loaded data.
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

    model = torchvision.models.AlexNet(num_classes=10)
    model.to(device)

    # model2 = copy.deepcopy(model1)

    pso = PSO(model=model, num_particles=2, inertia_weight=0.9,
              social_weight=0.5, cognitive_weight=0.8, min_param_value=-1,
              max_param_value=1, max_iterations=200, train_loader=valid_loader, device=device)
    # for now just use the validation set

    pso.optimize(visualize=True, output1="experiment1_loss.pt", output2="experiment1_accuracy.pt",
                 output3="experiment1_pca.pt")

    # final loss on test set
    loss, accuracy = evaluate_model(model, test_loader)
    print("final test loss: ", loss)
    print("final test accuracy: ", accuracy)

    #  weights = extract_weights(model)
    #  print(weights.shape)
    #  pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #  pytorch_total_params = sum(p.numel() for p in model.parameters())
    #  print("trainable: ", pytorch_total_trainable_params)
    #  print("total params: ", pytorch_total_params)

    # set_weights(model, weights + 1)
    # print(weights.shape)

    # for p1, p2 in zip(model.parameters(), model2.parameters()):
    #    print(torch.equal(p1, p2 + 1))
