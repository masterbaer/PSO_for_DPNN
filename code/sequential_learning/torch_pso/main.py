"""
This is our first try to use Particle Swarm Optimization (PSO) on neural network training.
We found a project called torch-pso (https://pypi.org/project/torch-pso/) which we simply use here.
We have do define a loss function (called closure()). We set it to the loss of the next batch.
Every time the loss is evaluated, a new batch is used to do so.
"""


import torch
import torchvision
from torch_pso import ParticleSwarmOptimizer

from dataloader import get_dataloaders_cifar10, set_all_seeds
from helper_train import train_model
import result_plotter

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ############
    # SETTINGS #
    ############

    seed = 123  # Set random seed.
    b = 256  # Set batch size.
    e = 20  # Set number of epochs to be trained.

    # Get device used for training, e.g., check via torch.cuda.is_available().
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    print(f'Using {device} device.')
    set_all_seeds(seed)  # Set all seeds to chosen random seed.

    ###########
    # DATASET #
    ###########

    # Using transforms on your data allows you to transform it from its
    # source state so that it is ready for training/validation/testing.

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
        print('Class labels of 10 examples:', labels[:10])
        break

    ########
    # MAIN #
    ########

    model = torchvision.models.AlexNet(num_classes=10)
    model = model.to(device)

    # The parameters max_param_value and min_param_value are only used for the initialization.
    # The weights are picked randomly in the interval [min,max]
    # The initial velocities are picked randomly in the interval [min-max, max-min]

    optimizer = ParticleSwarmOptimizer(model.parameters(),
                                       inertial_weight=0.5,
                                       num_particles=4,
                                       max_param_value=1,
                                       min_param_value=-1)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max', verbose=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss()

    # TRAIN MODEL.
    loss_list, train_acc_list, valid_acc_list = train_model(model=model, num_epochs=e,
                                                            train_loader=train_loader, valid_loader=valid_loader,
                                                            test_loader=test_loader,  # scheduler=scheduler,
                                                            optimizer=optimizer, criterion=criterion, device=device,
                                                            logging_interval=100)

    # Save history lists for loss, training accuracy, and validation accuracy to files.
    torch.save(loss_list, 'torch_pso_loss.pt')
    torch.save(train_acc_list, 'torch_pso_train_acc.pt')
    torch.save(valid_acc_list, 'torch_pso_valid_acc.pt')
    result_plotter.save_loss(loss_list)
    result_plotter.save_train_acc(train_acc_list)
    result_plotter.save_valid_acc(valid_acc_list)
