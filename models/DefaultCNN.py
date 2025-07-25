"""
Working Zombie versus Human Classifier!
- Updated CNN for 0-2 people per image classification with overfitting prevention
- Multi-task learning: person count + classification for each detected person
- Based on the pytorch example here: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random


class DefaultCNN(nn.Module):
    """
    Multi-task CNN for 0-2 people classification with overfitting prevention
    Outputs: [person_count, person1_class_probs, person2_class_probs]
    """

    def __init__(self, num_classes_=4, input_size=512, max_people=2, legacy_mode=False):
        super(DefaultCNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes_
        self.max_people = max_people
        self.legacy_mode = legacy_mode  # For backward compatibility with old models

        if legacy_mode:
            # Legacy architecture for old baseline.pth models
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=2, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.drop = nn.Dropout2d(p=0.2)
            
            # We'll calculate the FC layer size dynamically in the first forward pass
            self.fc = None
            self.feature_size = None
        else:
            # Enhanced architecture with more regularization
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Always output 4x4 regardless of input size
            
            # Increased dropout for overfitting prevention
            self.drop = nn.Dropout2d(p=0.3)
            self.drop_fc = nn.Dropout(p=0.5)
            
            # Batch normalization for better training stability
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)

            # Multi-task heads
            self.feature_size = 64 * 4 * 4  # After adaptive pooling
            
            # Person count head (0, 1, or 2 people)
            self.count_head = nn.Sequential(
                nn.Linear(self.feature_size, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 3)  # 0, 1, or 2 people
            )
            
            # Classification heads for each possible person
            self.person1_head = nn.Sequential(
                nn.Linear(self.feature_size, 128),
                nn.ReLU(), 
                nn.Dropout(0.4),
                nn.Linear(128, self.num_classes)
            )
            
            self.person2_head = nn.Sequential(
                nn.Linear(self.feature_size, 128),
                nn.ReLU(),
                nn.Dropout(0.4), 
                nn.Linear(128, self.num_classes)
            )

    def _get_conv_output_size(self, shape):
        """Calculate the output size after convolutions and pooling (legacy mode)"""
        batch_size = 1
        dummy_input = torch.zeros(batch_size, *shape)
        
        x = F.relu(self.pool(self.conv1(dummy_input)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.dropout(self.drop(x), training=self.training)
        
        return x.view(batch_size, -1).size(1)

    def forward(self, x):
        if self.legacy_mode:
            # Legacy forward pass for old models
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
        else:
            # Enhanced multi-task forward pass
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = self.drop(x)
            
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = self.drop(x)
            
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.adaptive_pool(x)  # Consistent output size
            x = self.drop(x)
            
            # Flatten for fully connected layers
            x = x.view(x.size(0), -1)
            x = self.drop_fc(x)
            
            # Multi-task outputs
            count_logits = self.count_head(x)
            person1_logits = self.person1_head(x)
            person2_logits = self.person2_head(x)
            
            return {
                'count': count_logits,
                'person1': person1_logits,
                'person2': person2_logits
            }

    def get_probs(self, logits_dict):
        """Convert logits to probabilities for RL interface"""
        with torch.no_grad():
            count_probs = F.softmax(logits_dict['count'], dim=1)
            person1_probs = F.softmax(logits_dict['person1'], dim=1)
            person2_probs = F.softmax(logits_dict['person2'], dim=1)
            
            # Weight person probabilities by count probability
            # If count says 0 people, zero out person probs
            # If count says 1 person, zero out person2 probs, etc.
            count_dist = count_probs[0]  # Assuming batch size 1
            
            # Weighted average approach
            final_person1 = person1_probs[0] * (count_dist[1] + count_dist[2])  # 1 or 2 people
            final_person2 = person2_probs[0] * count_dist[2]  # Only if 2 people
            
            return final_person1.numpy(), final_person2.numpy(), count_dist.numpy()
