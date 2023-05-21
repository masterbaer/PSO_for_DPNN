import torch
import torchvision

from sequential_learning.particle_swarm_opt.PSO import PSO
from sequential_learning.particle_swarm_opt.dataloader import get_dataloaders_cifar10

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

    pso.optimize()

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
