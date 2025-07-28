#!/usr/bin/env python3
"""
Status Model Evaluation Script
Tests the accuracy of the status model on the real dataset
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class TransferStatusCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(TransferStatusCNN, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        
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

def load_status_model(model_path='models/transfer_status_baseline.pth'):
    """Load the status model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Load model
    model = TransferStatusCNN(num_classes=4)  # 4 status classes: healthy, injured, zombie, corpse
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Status model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load status model: {e}")
        return None
    
    model.eval()
    return model, device

def preprocess_image(image_path, device):
    """Preprocess image for model input"""
    try:
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def create_status_label(row):
    """Create status label from metadata"""
    class_val = row.get('Class', 'Default')
    injured_val = row.get('Injured', 'False')
    
    if str(class_val).lower() == 'zombie':
        return 'zombie'  # All zombies are just 'zombie', regardless of injured status
    else:
        return 'injured' if str(injured_val).lower() == 'true' else 'healthy'

def evaluate_status_model():
    print("🔍 Status Model Evaluation")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Load model
    model = TransferStatusCNN(num_classes=3)  # Changed to 3 classes
    model_path = 'models/transfer_status_baseline.pth'
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Status model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load status model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Load metadata
    metadata_path = 'data/metadata.csv'
    df = pd.read_csv(metadata_path)
    print(f"📊 Loaded {len(df)} samples from metadata")
    
    # Filter out beaver images and empty entries
    df_filtered = df[
        (~df['Filename'].str.contains('beaver', case=False, na=False)) &
        (df['Class'].notna()) &
        (df['Class'] != '')
    ].copy()
    
    print(f"📊 Filtered to {len(df_filtered)} samples (excluding beaver)")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process images
    print("🔄 Processing images...")
    all_predictions = []
    all_labels = []
    processed_count = 0
    
    status_classes = ['healthy', 'injured', 'zombie']  # Only 3 classes
    
    for idx, row in df_filtered.iterrows():
        try:
            # Load image
            img_path = os.path.join('data', row['Filename'])
            if not os.path.exists(img_path):
                print(f"Error preprocessing image {row['Filename']}: [Errno 2] No such file or directory: '{img_path}'")
                continue
                
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                prediction = predicted.item()
            
            # Get true label
            status = create_status_label(row)
            true_label = status_classes.index(status)
            
            all_predictions.append(prediction)
            all_labels.append(true_label)
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"   Processed {processed_count}/{len(df_filtered)} images...")
                
        except Exception as e:
            print(f"Error preprocessing image {row['Filename']}: {e}")
            continue
    
    print(f"✅ Processed {processed_count} images successfully")
    
    # Calculate metrics
    print("\n📊 Evaluation Results:")
    print("=" * 50)
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=status_classes))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\n📊 Confusion Matrix:")
    print("         hea inj zom")
    for i, class_name in enumerate(status_classes):
        print(f"{class_name:8} {cm[i]}")
    
    # Calculate per-class accuracy
    print("\n📈 Per-class accuracy:")
    for i, class_name in enumerate(status_classes):
        if cm[i].sum() > 0:
            accuracy = cm[i][i] / cm[i].sum() * 100
            print(f"{class_name:8} - {accuracy:.1f}% ({cm[i][i]}/{cm[i].sum()})")
        else:
            print(f"{class_name:8} - 0.0% (0/0)")
    
    # Overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_predictions) * 100
    print(f"\n🎯 Overall accuracy: {overall_accuracy:.1f}%")
    
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=status_classes, yticklabels=status_classes)
    plt.title('Status Model Confusion Matrix (3-Class)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('status_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("💾 Confusion matrix saved as 'status_confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_status_model() 