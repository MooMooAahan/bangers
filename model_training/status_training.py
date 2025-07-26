#!/usr/bin/env python3
"""
Elegant Status-First CNN Training
4-class zombie detection with 70%+ accuracy target
Hierarchical approach: Master status first, add occupation later
"""

import pandas as pd
import os
import sys
import torch
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.utils
import sklearn.metrics
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

# Add parent directory to path
sys.path.append('../')
from models.DefaultCNN import DefaultCNN

print("🎯 Elegant Status-First CNN Training")
print("📊 Target: 70%+ accuracy on 4-class zombie detection")

# Load and balance dataset (reuse balancing strategy)
full_df = pd.read_csv('../data/metadata.csv')
single_person_df = full_df[full_df['HumanoidCount'] == 1].copy()

# Create 4-class status labels (ignore occupation)
def create_status_label(row):
    class_val = row.get('Class', 'Default')
    injured_val = row.get('Injured', 'False')
    
    if str(class_val).lower() == 'zombie':
        return 'corpse' if str(injured_val).lower() == 'true' else 'zombie'
    else:
        return 'injured' if str(injured_val).lower() == 'true' else 'healthy'

single_person_df['status'] = single_person_df.apply(create_status_label, axis=1)

# Smart balancing: Ensure roughly equal status distribution
print(f"📊 Original status distribution:")
status_counts = single_person_df['status'].value_counts()
print(status_counts)

# Balance by replicating minority classes
target_count = max(status_counts.values)
balanced_dfs = []

for status in status_counts.index:
    status_df = single_person_df[single_person_df['status'] == status].copy()
    current_count = len(status_df)
    
    if current_count < target_count:
        # Calculate replication needed
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
print(f"📊 Final status distribution:")
print(balanced_df['status'].value_counts())

# Encode status to integers
status_encoder = sklearn.preprocessing.LabelEncoder()
status_classes = ['healthy', 'injured', 'zombie', 'corpse']  # Logical order
status_encoder.classes_ = np.array(status_classes, dtype=object)
balanced_df['status_encoded'] = status_encoder.transform(balanced_df['status'])

# Train/test split
train_df, test_df = sklearn.model_selection.train_test_split(
    balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['status'])

print(f"\n📊 Train samples: {len(train_df)}")
print(f"📊 Test samples: {len(test_df)}")

# Strong augmentation for maximum data utilization
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
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
            return torch.zeros(3, 512, 512), torch.tensor(0).long()

# Setup training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
epochs = 20

train_ds = StatusDataset(train_df, train_transform)
val_ds = StatusDataset(test_df, val_transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

# 4-class model
model = DefaultCNN(num_classes_=4, input_size=512)
model.to(device)

# Balanced loss weighting
class_weights = sklearn.utils.class_weight.compute_class_weight(
    'balanced', classes=np.unique(train_df.status_encoded), y=train_df.status_encoded)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
print(f"\n🚀 Training 4-class status model on {device}...")
best_accuracy = 0
patience = 5
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
    scheduler.step()
    
    print(f"📊 Epoch {epoch+1}: Train_Loss={running_loss/len(train_loader):.4f}, "
          f"Val_Loss={val_loss:.4f}, Val_Acc={val_accuracy:.1f}%")
    
    # Save best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), '../models/status_baseline.pth')
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

print(f"\n💾 Status model saved as: models/status_baseline.pth")
print(f"🚀 Ready for hierarchical expansion: Add occupation classifier next!")

if best_accuracy >= 70:
    print(f"\n🎉 SUCCESS! Achieved {best_accuracy:.1f}% accuracy (target: 70%)")
    print("✅ Status detection mastered - foundation ready for occupation layer")
else:
    print(f"\n⚠️ Need improvement: {best_accuracy:.1f}% < 70% target")
    print("💡 Consider: More augmentation, different architecture, or more training data") 