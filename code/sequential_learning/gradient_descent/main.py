###########
# IMPORTS #
###########

import torch
import torchvision
from dataloader import get_dataloaders_cifar10, set_all_seeds
from helper_train import train_model
import result_plotter
from sequential_learning.gradient_descent.model import NeuralNetwork

"""
We use gradient descent with a batch size of 256 and learning rate of 0.01. 
Using a learning rate scheduler (ReduceLROnPlateau) leads to the same validation accuracy of 57% as not using one
but it increases the training accuracy (which is not useful).
"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ############
    # SETTINGS #
    ############

    seed = 123  # Set random seed.
    b = 256  # Set batch size.
    e = 40  # Set number of epochs to be trained.

    # Get device used for training, e.g., check via torch.cuda.is_available().
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    print(f'Using {device} device.')
    set_all_seeds(seed)  # Set all seeds to chosen random seed.

    ###########
    # DATASET #
    ###########

    # Using transforms on your data allows you to transform it from its
    # source state so that it is ready for training/validation/testing.

    cifar_10_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # GET PYTORCH DATALOADERS FOR TRAINING, TESTING, AND VALIDATION DATASET.
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        root="../../data",
        validation_fraction=0.1,
        train_transforms=cifar_10_transforms,
        test_transforms=cifar_10_transforms,
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

    # model = torchvision.models.AlexNet(num_classes=10)
    num_classes = 10
    image_shape = None
    for images, labels in train_loader:
        image_shape = images.shape
        break
    model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).cuda()

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
    #result_plotter.save_loss(loss_list)
    #result_plotter.save_train_acc(train_acc_list)
    #result_plotter.save_valid_acc(valid_acc_list)
