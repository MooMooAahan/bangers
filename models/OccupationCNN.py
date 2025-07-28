"""
Occupation CNN Model
CNN for classifying occupations: civilian, child, doctor, police, militant
Based on the successful DefaultCNN architecture but adapted for 5-class occupation classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OccupationCNN(nn.Module):
    """
    CNN for occupation classification: civilian, child, doctor, police, militant
    """

    def __init__(self, num_classes=5, input_size=512):
        super(OccupationCNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes

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

        # Compute the size of the FC layer if not already computed
        if self.fc is None:
            self.feature_size = self._get_conv_output_size(x.shape[1:])
            self.fc = nn.Linear(self.feature_size, self.num_classes).to(x.device)

        # Use the layers here with appropriate activation function
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))

        # Apply dropout
        x = F.dropout(self.drop(x), training=self.training)

        # Prep for linear layer
        # Flatten the output (except batch_size dim)
        x = x.view(-1, self.feature_size)

        # Fully connected layer
        x = self.fc(x)

        return x
