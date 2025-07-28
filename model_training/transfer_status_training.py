#!/usr/bin/env python3
"""
Transfer Status CNN Training Script
Trains a ResNet18-based model for 4-class status classification
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class TransferStatusCNN(nn.Module):
    def __init__(self, num_classes=4):
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

def create_status_label(row):
    """Create status label from metadata"""
    class_val = row.get('Class', 'Default')
    injured_val = row.get('Injured', 'False')
    
    if str(class_val).lower() == 'zombie':
        return 'zombie'  # All zombies are just 'zombie', regardless of injured status
    else:
        return 'injured' if str(injured_val).lower() == 'true' else 'healthy'

class StatusDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, base_dir='data'):
        self.df = df
        self.transform = transform
        self.base_dir = base_dir
        self.status_classes = ['healthy', 'injured', 'zombie']  # Only 3 classes
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.base_dir, row['Filename'])
        image = Image.open(img_path).convert('RGB')
        
        # Get status label
        status = create_status_label(row)
        label = self.status_classes.index(status)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label).long()

def train_model(model, train_loader, val_loader, num_epochs=20, device='cpu'):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
    model.train()
    running_loss = 0.0
        train_correct = 0
        train_total = 0
    
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/transfer_status_baseline.pth')
            print(f"💾 New best model! Validation accuracy: {val_acc:.2f}%")
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report
    status_classes = ['healthy', 'injured', 'zombie']
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=status_classes))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=status_classes, yticklabels=status_classes)
    plt.title('Status Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('model_training/status_model_test_results.png', dpi=300, bbox_inches='tight')
    print("💾 Confusion matrix saved as 'model_training/status_model_test_results.png'")

def main():
    print("🏥 Transfer Status CNN Training")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Load metadata
    metadata_path = 'data/metadata.csv'
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found: {metadata_path}")
        return
    
    df = pd.read_csv(metadata_path)
    print(f"📊 Loaded {len(df)} samples from metadata")
    
    # Filter out beaver images and empty entries
    df_filtered = df[
        (~df['Filename'].str.contains('beaver', case=False, na=False)) &
        (df['Class'].notna()) &
        (df['Class'] != '')
    ].copy()
    
    print(f"📊 Filtered to {len(df_filtered)} samples")
    
    # Create status labels
    df_filtered['status'] = df_filtered.apply(create_status_label, axis=1)
    
    # Check status distribution
    status_counts = df_filtered['status'].value_counts()
    print("\n📈 Status distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Split data
    train_df, temp_df = train_test_split(df_filtered, test_size=0.3, random_state=42, stratify=df_filtered['status'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['status'])
    
    print(f"\n📊 Data split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = StatusDataset(train_df, train_transform)
    val_dataset = StatusDataset(val_df, val_transform)
    test_dataset = StatusDataset(test_df, val_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create model
    model = TransferStatusCNN(num_classes=3) # Changed to 3 classes
    model.to(device)
    
    # Train model
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, num_epochs=20, device=device)
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    evaluate_model(model, test_loader, device)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('model_training/status_training_curves.png', dpi=300, bbox_inches='tight')
    print("💾 Training curves saved as 'model_training/status_training_curves.png'")
    
    print(f"\n✅ Training completed! Model saved as 'models/transfer_status_baseline.pth'")

if __name__ == "__main__":
    main() 