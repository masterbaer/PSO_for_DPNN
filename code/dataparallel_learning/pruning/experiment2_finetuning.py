"""
Here we test if the models can improve with finetuning.
"""
import torch
import torchvision

from dataloader import set_all_seeds, get_dataloaders_cifar10
from model import NeuralNetwork, CombinedNeuralNetwork


def get_number_of_zero_params(model):
    number_of_zeros = 0
    number_of_params = 0

    for i, layer in enumerate(model.modules()):
        number_of_zeros_local = 0
        number_of_params_local = 0

        if isinstance(layer, torch.nn.Linear):

            for param in layer.parameters():
                # count number of zeros
                # https://discuss.pytorch.org/t/how-to-count-the-number-of-zero-weights-in-a-pytorch-model/13549/2
                number_of_zeros_local += param.numel() - param.nonzero().size(0)
                number_of_params_local += param.numel()

        number_of_zeros += number_of_zeros_local
        number_of_params += number_of_params_local

    return number_of_zeros, number_of_params

def evaluate_model(model, data_loader, device):
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
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
        test_loss = test_loss + loss.item()

    loss_per_batch = test_loss / number_of_batches
    accuracy = (correct_pred.float() / num_examples).item()
    return loss_per_batch, accuracy

if __name__ == '__main__':

    seed = 0
    set_all_seeds(seed)

    b = 256
    e = 30

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    set_all_seeds(seed)  # Set all seeds to chosen random seed.

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

    learning_rate = 0.01

    num_classes = 10
    image_shape = None
    for images, labels in train_loader:
        image_shape = images.shape
        break





    model0 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    model1 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    model2 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    model3 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    combined_model = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    combined_model_scratch = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)

    #model0.load_state_dict(torch.load("ex1_model_0.pt"))
    #model1.load_state_dict(torch.load("ex1_model_1.pt"))
    #model2.load_state_dict(torch.load("ex1_model_2.pt"))
    #model3.load_state_dict(torch.load("ex1_model_3.pt"))
    combined_model.load_state_dict(torch.load("ex1_combined_model.pt"))
    combined_model_scratch.load_state_dict(torch.load("ex1_combined_model_scratch.pt"))

    optimizer_combined = torch.optim.SGD(combined_model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_combined_scratch = torch.optim.SGD(combined_model_scratch.parameters(), lr=learning_rate, momentum=0.9)

    test_accuracy_list_combined = []
    test_accuracy_list_scratch = []

    _, test_accuracy_combined = evaluate_model(combined_model, test_loader, device)
    print(f"ensemble accuracy before finetuning: ", test_accuracy_combined)

    _, test_accuracy_scratch = evaluate_model(combined_model_scratch, test_loader, device)
    print(f"large model accuracy before finetuning: ", test_accuracy_scratch)

    test_accuracy_list_scratch.append(test_accuracy_scratch)
    test_accuracy_list_combined.append(test_accuracy_combined)

    for epoch in range(e):

        #model0.train()
        combined_model.train()
        combined_model_scratch.train()

        for i, (inputs, labels) in enumerate(train_loader):

            #train the combined model
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_combined.zero_grad()
            outputs_combined = combined_model(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs_combined, labels)
            loss.backward()
            optimizer_combined.step()

            optimizer_combined_scratch.zero_grad()
            outputs_combined_scratch = combined_model_scratch(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss_combined_scratch = loss_fn(outputs_combined_scratch, labels)
            loss_combined_scratch.backward()
            optimizer_combined_scratch.step()

        combined_model.eval()
        combined_model_scratch.eval()

        _, test_accuracy = evaluate_model(combined_model, test_loader, device)
        print(f"combined model : test accuracy after {epoch+1} epochs: {test_accuracy}")

        _, test_accuracy_scratch = evaluate_model(combined_model_scratch, test_loader, device)
        print(f"combined model scratch : test accuracy after {epoch + 1} epochs: {test_accuracy_scratch}")

        test_accuracy_list_combined.append(test_accuracy)
        test_accuracy_list_scratch.append(test_accuracy_scratch)

    combined_model.eval()
    _, accuracy_combined = evaluate_model(combined_model, test_loader, device)
    print("accuracy_combined")

    torch.save(test_accuracy_list_scratch, 'ex2_combined_scratch_test_accuracy.pt')
    torch.save(test_accuracy_list_combined, 'ex2_combined_test_accuracy.pt')

    num_zeros, num_params = get_number_of_zero_params(combined_model)
    print(f"{num_zeros} parameters out of  {num_params} are zero")


    #torch.save(test_accuracy_list, f'ex2_test_accuracy{0}.pt')
