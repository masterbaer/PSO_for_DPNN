#########
# MODEL #
#########

import torch


class AlexNet(torch.nn.Module):

    def __init__(self, num_classes=1000, dropout=0.5):  # Initialize layers in __init__.

        super().__init__()

        self.features = torch.nn.Sequential(  # feature-extractor part
            # 1st convolutional layer (+ max-pooling)
            torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            # 2nd convolutional layer (+ max-pooling)
            torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            # 3rd + 4th convolutional layer
            torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),

            # 5th convolutional layer (+ max-pooling)
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Average pooling to downscale possibly larger input images.
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = torch.nn.Sequential(  # fully-connected part

            # 6th, 7th + 8th fully connected layer
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(256 * 6 * 6, 4096),  # 6th FC layer with ‘in_features‘ + ‘out_features‘
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(4096, 4096),  # 7th FC layer
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes)  # 8th FC layer with probabilities of belonging to each class
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
