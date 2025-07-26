#!/usr/bin/env python3
"""
Elegant Transfer Learning Status CNN
Using ResNet18 pretrained backbone for 70%+ accuracy
The smart, proven approach to computer vision
"""

import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.utils
import sklearn.metrics
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

print("🎯 Elegant Transfer Learning Status CNN")
print("📊 Using ResNet18 pretrained backbone for 70%+ accuracy")

# Load and balance dataset
full_df = pd.read_csv('../data/metadata.csv')
single_person_df = full_df[full_df['HumanoidCount'] == 1].copy()

# Create 4-class status labels
def create_status_label(row):
    class_val = row.get('Class', 'Default')
    injured_val = row.get('Injured', 'False')
    
    if str(class_val).lower() == 'zombie':
        return 'corpse' if str(injured_val).lower() == 'true' else 'zombie'
    else:
        return 'injured' if str(injured_val).lower() == 'true' else 'healthy'

single_person_df['status'] = single_person_df.apply(create_status_label, axis=1)

# Smart balancing
status_counts = single_person_df['status'].value_counts()
print(f"📊 Original status distribution:")
print(status_counts)

target_count = max(status_counts.values)
balanced_dfs = []

for status in status_counts.index:
    status_df = single_person_df[single_person_df['status'] == status].copy()
    current_count = len(status_df)
    
    if current_count < target_count:
        multiplier = target_count // current_count
        remainder = target_count % current_count
        
        replicated_dfs = [status_df] * multiplier
        if remainder > 0:
            replicated_dfs.append(status_df.sample(n=remainder, random_state=42))
        
        balanced_status_df = pd.concat(replicated_dfs, ignore_index=True)
        print(f"  {status.capitalize()}: {current_count} → {len(balanced_status_df)}")
    else:
        balanced_status_df = status_df
        print(f"  {status.capitalize()}: {current_count} (no change)")
    
    balanced_dfs.append(balanced_status_df)

balanced_df = pd.concat(balanced_dfs, ignore_index=True)
print(f"\n✅ Balanced dataset: {len(balanced_df)} images")

# Encode status
status_classes = ['healthy', 'injured', 'zombie', 'corpse']
status_encoder = sklearn.preprocessing.LabelEncoder()
status_encoder.classes_ = np.array(status_classes, dtype=object)
balanced_df['status_encoded'] = status_encoder.transform(balanced_df['status'])

# Train/test split
train_df, test_df = sklearn.model_selection.train_test_split(
    balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['status'])

print(f"📊 Train samples: {len(train_df)}")
print(f"📊 Test samples: {len(test_df)}")

# ImageNet-style transforms for transfer learning
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet standard input size
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class StatusDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms, base_dir="../data"):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.base_dir = base_dir
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = row['Filename']
        status_label = row['status_encoded']
        
        try:
            image = Image.open(os.path.join(self.base_dir, img_path))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.transforms(image)
            return image, torch.tensor(status_label).long()
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(0).long()

# Transfer Learning Model
class TransferStatusCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(TransferStatusCNN, self).__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers (optional - can unfreeze for fine-tuning)
        for param in list(self.backbone.parameters())[:-20]:  # Freeze most layers
            param.requires_grad = False
            
        # Replace final classification layer
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

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32  # Can use larger batch with efficient ResNet
epochs = 25

train_ds = StatusDataset(train_df, train_transform)
val_ds = StatusDataset(test_df, val_transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

# Transfer learning model
model = TransferStatusCNN(num_classes=4)
model.to(device)

print(f"🧠 Created transfer learning model with ResNet18 backbone")
print(f"🔥 Training on {device} with batch size {batch_size}")

# Optimized training setup
class_weights = sklearn.utils.class_weight.compute_class_weight(
    'balanced', classes=np.unique(train_df.status_encoded), y=train_df.status_encoded)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Lower learning rate for transfer learning
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3)

def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    
    return accuracy, avg_loss, all_preds, all_labels

# Training loop
print(f"\n🚀 Training transfer learning model...")
best_accuracy = 0
patience = 7
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation
    val_accuracy, val_loss, val_preds, val_labels = evaluate_model(model, val_loader)
    scheduler.step(val_loss)
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"📊 Epoch {epoch+1}: Train_Loss={avg_train_loss:.4f}, "
          f"Val_Loss={val_loss:.4f}, Val_Acc={val_accuracy:.1f}%")
    
    # Save best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), '../models/transfer_status_baseline.pth')
        print(f"   ✅ New best model saved! Accuracy: {val_accuracy:.1f}%")
        patience_counter = 0
    else:
        patience_counter += 1
        
    # Early stopping
    if patience_counter >= patience:
        print(f"   ⏹️ Early stopping triggered (patience={patience})")
        break

print(f"\n🎉 Training completed!")
print(f"🏆 Best accuracy achieved: {best_accuracy:.1f}%")
print(f"🎯 Target was 70% - {'✅ SUCCESS!' if best_accuracy >= 70 else '❌ Need improvement'}")

# Final evaluation
final_accuracy, _, final_preds, final_labels = evaluate_model(model, val_loader)

# Detailed results
print(f"\n📊 Final Detailed Results:")
print(f"🎯 Final Accuracy: {final_accuracy:.1f}%")

# Confusion matrix
cm = sklearn.metrics.confusion_matrix(final_labels, final_preds)
print(f"\n🔍 Confusion Matrix:")
print("     Pred:", " ".join([f"{cls:>8}" for cls in status_classes]))
for i, true_class in enumerate(status_classes):
    print(f"True {true_class:>8}:", " ".join([f"{cm[i,j]:>8}" for j in range(len(status_classes))]))

# Classification report
report = sklearn.metrics.classification_report(
    final_labels, final_preds, 
    target_names=status_classes,
    zero_division=0
)
print(f"\n📋 Classification Report:")
print(report)

# Per-class accuracy
print(f"\n🎯 Per-Class Accuracy:")
for i, class_name in enumerate(status_classes):
    class_mask = np.array(final_labels) == i
    if class_mask.sum() > 0:
        class_acc = (np.array(final_preds)[class_mask] == i).mean() * 100
        print(f"  {class_name.capitalize()}: {class_acc:.1f}%")

print(f"\n💾 Transfer learning model saved as: models/transfer_status_baseline.pth")

if best_accuracy >= 70:
    print(f"\n🎉 MISSION ACCOMPLISHED! 🎉")
    print(f"✅ Achieved {best_accuracy:.1f}% accuracy (target: 70%)")
    print("🚀 Status detection mastered with transfer learning!")
    print("📈 Ready for hierarchical expansion: Add occupation classifier next!")
else:
    print(f"\n⚠️ Still improving: {best_accuracy:.1f}% < 70% target")
    print("💡 Transfer learning significantly better than custom CNN")
    print("🔧 Consider: Fine-tuning more layers or different backbone") 