"""
Working Zombie versus Human Classifier!
- This a basic CNN implemented in pytorch for predicting if an image is a Human or a Zombie.
- Based on the pytorch example here: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DefaultCNN(nn.Module):

    # Defining the Constructor
    def __init__(self, num_classes_=4, input_size=512):
        # In the init function, we define each layer we will use in our model
        super(DefaultCNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes_

        # Our images are RGB, so we have input channels = 3.
        # We will apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=2, stride=1, padding=1)

        # A second convolutional layer takes 12 input channels, and generates 24 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)

        # We in the end apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)

        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)

        # We'll calculate the FC layer size dynamically in the first forward pass
        self.fc = None
        self.feature_size = None

    def _get_conv_output_size(self, shape):
        """Calculate the output size after convolutions and pooling"""
        batch_size = 1
        dummy_input = torch.zeros(batch_size, *shape)
        
        x = F.relu(self.pool(self.conv1(dummy_input)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.dropout(self.drop(x), training=self.training)
        
        return x.view(batch_size, -1).size(1)

    def forward(self, x):
        # In the forward function, pass the data through the layers we defined in the init function
        
        # Initialize the fully connected layer if not done yet
        if self.fc is None:
            self.feature_size = self._get_conv_output_size(x.shape[1:])
            self.fc = nn.Linear(in_features=self.feature_size, out_features=self.num_classes)
            # Move to the same device as input
            self.fc = self.fc.to(x.device)

        # Use a ReLU activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))

        # Use a ReLU activation function after layer 2
        x = F.relu(self.pool(self.conv2(x)))

        # Select some features to drop to prevent overfitting (only drop during training)
        x = F.dropout(self.drop(x), training=self.training)

        # Flatten - calculate size dynamically to handle any input size
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        return x
