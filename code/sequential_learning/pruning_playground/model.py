#  This file contains neural network models used in our experiments.
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, output_dim)

        # self.linear_relu_stack = nn.Sequential(
        #    nn.Linear(input_dim, 512),
        #    nn.ReLU(),
        #    nn.Linear(512, 512),
        #    nn.ReLU(),
        #    nn.Linear(512, output_dim),
        # )

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)

        # logits = self.linear_relu_stack(x)
        return x  # logits


class CombinedNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512 * 4)
        self.fc2 = nn.Linear(512 * 4, 512 * 4)
        self.fc3 = nn.Linear(512 * 4, 512 * 4)
        self.fc4 = nn.Linear(512 * 4, output_dim)
        # divide fc outputs by 4 when creating the model

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


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


# https://github.com/soapisnotfat/pytorch-cifar10/blob/master/models/LeNet.py
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6,
                               kernel_size=5)  # 3 input channels,6 output feature maps, kernelsize 5 (stride 1, padding 0 implicitly)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    """
    (see # https://www.baeldung.com/cs/convolutional-layer-size: 
        conv2d: (output_size = 1 + (input_size - kernel_size + 2 * padding) / stride) )
        pooling: half for 2x2 masks
        
    Input image: [3,32,32]. 
    after conv1: [6, 28, 28]                     1+(32-5)/1 = 28
    after pooling1: [6,14,14]                 
    after conv2: [16,10, 10]                     1+(14-5)/1 = 10
    after pooling2: [16,5,5]
    after flattening: [16*5*5] = [120]
    after fc2: [84]
    after fc3: [10]
    """


class CombinedLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CombinedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6 * 4,
                               kernel_size=5)  # 3 input channels,6 output feature maps, kernelsize 5 (stride 1, padding 0 implicitly)
        self.conv2 = nn.Conv2d(6 * 4, 16 * 4, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5 * 4, 120 * 4)
        self.fc2 = nn.Linear(120 * 4, 84 * 4)
        self.fc3 = nn.Linear(84 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    # 'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # Input image: [3,32,32]

    # conv1: [64, 32, 32]         1+(32-3+2*1)/1 = 32
    # maxpool1 : [64, 16, 16]
    # conv2: [128, 16,16]
    # maxpool2: [128,8,8]
    # conv3: [256,8,8]
    # conv4: [256,8,8]
    # maxpool3: [256,4,4]
    # conv5 [512,4,4]
    # conv6 [512,4,4]
    # maxpool4 [512,2,2]
    # conv7 [512,2,2]
    # conv8 [512,2,2]
    # maxpool5 [512,1,1]
    # flatten --> [512]





class CombinedVGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(CombinedVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, 4 * x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(4 * x),
                           nn.ReLU(inplace=True)]
                in_channels = 4 * x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    # 'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # Input image: [3,32,32]

    # conv1: [64*4, 32, 32]
    # maxpool1 : [64*4, 16, 16]
    # conv2: [128*4, 16,16]
    # maxpool2: [128*4,8,8]
    # conv3: [256*4,8,8]
    # conv4: [256*4,8,8]
    # maxpool3: [256*4,4,4]
    # conv5 [512*4,4,4]
    # conv6 [512*4,4,4]
    # maxpool4 [512*4,2,2]
    # conv7 [512*4,2,2]
    # conv8 [512*4,2,2]
    # maxpool5 [512*4,1,1]
    # flatten --> [512*4]


# This is can also be used for combining but torch-pruning cannot cope with 4 sequential BatchNorm-layers so we do not
# use this approach
"""
class CustomBatchNorm(nn.Module):
    def __init__(self, num_channels):
        super(CustomBatchNorm, self).__init__()

        self.num_channels = num_channels
        self.k = int(num_channels / 4)
        self.bn_1 = nn.BatchNorm2d(self.k)
        self.bn_2 = nn.BatchNorm2d(self.k)
        self.bn_3 = nn.BatchNorm2d(self.k)
        self.bn_4 = nn.BatchNorm2d(self.k)

    def forward(self, x):
        # Split the input
        x_1 = x[:, :self.k, :, :]
        x_2 = x[:, self.k:2 * self.k, :, :]
        x_3 = x[:, 2 * self.k:3 * self.k, :, :]
        x_4 = x[:, 3 * self.k:4 * self.k, :, :]

        # Apply batch normalization
        x1 = self.bn_1(x_1)
        x2 = self.bn_2(x_2)
        x3 = self.bn_3(x_3)
        x4 = self.bn_4(x_4)

        # Concatenate the batch-normalized group with the untouched group
        x = torch.cat([x1,x2,x3,x4], dim=1)
        return x
    
class CombinedVGGCustomBatch(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(CombinedVGGCustomBatch, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, 4 * x, kernel_size=3, padding=1),
                           CustomBatchNorm(4 * x), # this here is different
                           nn.ReLU(inplace=True)]
                in_channels = 4 * x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
"""