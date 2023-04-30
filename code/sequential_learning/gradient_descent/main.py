###########
# IMPORTS #
###########

import torch
import torchvision
from Alexnet import AlexNet
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
    e = 40  # Set number of epochs to be trained. Standard: 200 epochs

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

    model = AlexNet(num_classes=10)  # Build instance of AlexNet with 10 classes for CIFAR-10.
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # torch.optim.lr_scheduler provides several learning-rate adjustment methods based on number of epochs.
    # torch.optim.lr_scheduler.ReduceLROnPlateau: dynamic learning rate reducing based on some validation measurements.
    # Reduce learning rate when a metric has stopped improving:

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max', verbose=True)

    # TRAIN MODEL.
    loss_list, train_acc_list, valid_acc_list = train_model(model=model, num_epochs=e,
                                                            train_loader=train_loader, valid_loader=valid_loader,
                                                            test_loader=test_loader,
                                                            optimizer=optimizer, device=device,
                                                            scheduler=scheduler, logging_interval=100)

    # Save history lists for loss, training accuracy, and validation accuracy to files.
    torch.save(loss_list, 'seq_gd_loss.pt')
    torch.save(train_acc_list, 'seq_gd_train_acc.pt')
    torch.save(valid_acc_list, 'seq_gd_valid_acc.pt')
    result_plotter.save_loss(loss_list)
    result_plotter.save_train_acc(train_acc_list)
    result_plotter.save_valid_acc(valid_acc_list)
