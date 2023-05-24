#  This file contains neural network models used in our experiments.
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()

        self.features = torch.nn.Sequential(  # FEATURE-EXTRACTOR PART
            torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            # 1st convolutional layer with 3 input and 96 output channels
            torch.nn.ReLU(inplace=True),  # Rectified Linear Unit activation function
            torch.nn.MaxPool2d(kernel_size=3, stride=2),  # max-pooling
            torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),  # 2nd convolutional layer (+ max-pooling)
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 3rd convolutional layer
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),  # 4th convolutional layer
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 5th convolutional layer (+ max-pooling)
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2))

        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = torch.nn.Sequential(  # FULLY-CONNECTED (FC) MULTI-LAYER PERCEPTRON PART
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(256 * 6 * 6, 4096),  # 6th FC layer with ‘in_features‘ + ‘out_features‘
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(4096, 4096),  # 7th FC layer
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes))  # 8th FC layer with probabilities of belonging to each

    def forward(self, x):
        x = self.features(x)  # convolutional feature-extractor part
        x = self.avgpool(x)  # average pooling
        x = torch.flatten(x, 1)  # flattening
        x = self.classifier(x)  # classification
        return x
