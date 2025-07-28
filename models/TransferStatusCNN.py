#!/usr/bin/env python3
"""
Transfer Status CNN Model
ResNet18-based model for 3-class status classification
"""

import torch
import torch.nn as nn
from torchvision import models

class TransferStatusCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TransferStatusCNN, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
            
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x) 