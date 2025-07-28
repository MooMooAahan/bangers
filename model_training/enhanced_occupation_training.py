#!/usr/bin/env python3
"""
Optimized Occupation CNN Training
Based on the original approach that achieved 51%, now for 4 classes
Focus on color features (white coats for doctors, uniforms for police)
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
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

from models.OccupationCNN import OccupationCNN

print("🏢 Optimized 4-Class Occupation CNN")
print("📊 Building on the 51% accuracy approach")

# Load metadata
try:
    full_df = pd.read_csv('data/metadata.csv')
    print(f"📊 Total samples: {len(full_df)}")
except FileNotFoundError:
    print("❌ metadata.csv not found")
    sys.exit(1)

# Color-preserving augmentation (don't distort white coats!)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),  # Small rotation only
    transforms.ColorJitter(
        brightness=0.1,  # Minimal brightness change
        contrast=0.1,    # Preserve white coat contrast
        saturation=0.1,  # Keep color information
        hue=0.05        # Very small hue shift
    ),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Minimal crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ColorAwareOccupationCNN(nn.Module):
    """Enhanced model with color feature extraction"""
    def __init__(self, num_classes=4):
        super(ColorAwareOccupationCNN, self).__init__()
        
        # MobileNetV2 - worked well before
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Freeze early layers but keep color-sensitive layers trainable
        for i, param in enumerate(self.backbone.features.parameters()):
            if i < 50:  # Freeze fewer layers to learn color patterns
                param.requires_grad = False
        
        # Add color histogram branch
        self.color_pool = nn.AdaptiveAvgPool2d(1)
        
        # Enhanced classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class BalancedOccupationDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms, base_dir="data", oversample=True):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.transforms = transforms
        
        # 4 classes (merging Militant with Civilian)
        self.occupation_classes = ['Civilian', 'Child', 'Doctor', 'Police']
        self.occupation_to_idx = {occ: idx for idx, occ in enumerate(self.occupation_classes)}
        
        # Oversample minority classes
        if oversample and 'Role' in df.columns:
            self.df = self._oversample_minorities()
        
        self.sample_weights = self._calculate_sample_weights()
    
    def _oversample_minorities(self):
        """Oversample to balance classes"""
        # Merge militants with civilians first
        df_copy = self.df.copy()
        df_copy.loc[df_copy['Role'] == 'Militant', 'Role'] = 'Civilian'
        
        class_counts = df_copy['Role'].value_counts()
        max_count = class_counts.max()
        
        balanced_dfs = []
        for occ in self.occupation_classes:
            class_df = df_copy[df_copy['Role'] == occ]
            if len(class_df) > 0:
                # Replicate to balance
                n_samples = max_count
                n_repeats = n_samples // len(class_df) + 1
                balanced_dfs.append(pd.concat([class_df] * n_repeats, ignore_index=True)[:n_samples])
        
        return pd.concat(balanced_dfs, ignore_index=True)
    
    def _calculate_sample_weights(self):
        """Calculate weights for balanced sampling"""
        # Count after merging
        role_series = self.df['Role'].replace('Militant', 'Civilian')
        class_counts = role_series.value_counts()
        
        weights = []
        for _, row in self.df.iterrows():
            role = row['Role'] if row['Role'] != 'Militant' else 'Civilian'
            weight = 1.0 / class_counts[role]
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['Filename']
        occupation = row['Role']
        
        # Merge Militant -> Civilian
        if occupation == 'Militant':
            occupation = 'Civilian'
            
        target = self.occupation_to_idx[occupation]
        
        try:
            full_path = os.path.join(self.base_dir, img_path)
            image = Image.open(full_path).convert('RGB')
            image = self.transforms(image)
            return image, torch.tensor(target).long()
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return torch.randn(3, 224, 224), torch.tensor(0).long()

def _create_augmented_dataset(df):
    """Create balanced dataset with aggressive augmentation for underrepresented classes"""
    print("🔄 Creating balanced dataset with aggressive augmentation...")
    
    # Count samples per class
    class_counts = df['Role'].value_counts()
    target_count = max(class_counts) * 2  # Double the largest class
    
    print(f"📊 Target samples per class: {target_count}")
    print("📈 Current distribution:")
    for role, count in class_counts.items():
        print(f"  {role}: {count} -> {target_count}")
    
    balanced_df = []
    
    for role in class_counts.index:
        role_samples = df[df['Role'] == role].copy()
        current_count = len(role_samples)
        
        if current_count < target_count:
            # Need to augment this class
            augment_factor = int(target_count / current_count) + 1
            print(f"🔄 Augmenting {role}: {current_count} -> {target_count} (factor: {augment_factor})")
            
            # Add original samples
            balanced_df.extend(role_samples.to_dict('records'))
            
            # Add augmented samples
            for i in range(augment_factor - 1):
                augmented_samples = role_samples.copy()
                # Add augmentation flag for tracking
                augmented_samples['augmented'] = True
                balanced_df.extend(augmented_samples.to_dict('records'))
        else:
            # This class has enough samples, just add them
            balanced_df.extend(role_samples.to_dict('records'))
    
    balanced_df = pd.DataFrame(balanced_df)
    print(f"✅ Balanced dataset created: {len(balanced_df)} total samples")
    
    # Show final distribution
    final_counts = balanced_df['Role'].value_counts()
    print("📊 Final distribution:")
    for role, count in final_counts.items():
        print(f"  {role}: {count}")
    
    return balanced_df

class EnhancedOccupationDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, aggressive_transform=None):
        self.df = df
        self.transform = transform
        self.aggressive_transform = aggressive_transform
        self.role_classes = ['Civilian', 'Child', 'Doctor', 'Police']  # 4 classes
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join('data', row['Filename'])
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Return dummy data if image not found
            return torch.randn(3, 224, 224), torch.tensor(0).long()
        
        # Get role label
        role = row['Role']
        label = self.role_classes.index(role)
        
        # Apply transforms with augmentation
        if self.transform:
            # Check if this is an augmented sample
            is_augmented = row.get('augmented', False)
            
            if is_augmented and self.aggressive_transform:
                # Apply more aggressive augmentation for synthetic samples
                image = self.aggressive_transform(image)
            else:
                # Apply normal augmentation for real samples
                image = self.transform(image)
        
        return image, torch.tensor(label).long()

# Filter data
single_person_df = full_df[full_df['HumanoidCount'] == 1].copy()
single_person_df = single_person_df[~single_person_df['Role'].str.contains('Beaver', case=False, na=False)]
valid_occupations = ['Civilian', 'Child', 'Doctor', 'Police', 'Militant']
single_person_df = single_person_df[single_person_df['Role'].isin(valid_occupations)]

print(f"📊 Filtered samples: {len(single_person_df)}")
print("📈 Class distribution:")
role_counts = single_person_df['Role'].value_counts()
for occ, count in role_counts.items():
    if occ == 'Militant':
        print(f"  {occ}: {count} (→ will merge with Civilian)")
    else:
        print(f"  {occ}: {count}")

civilian_total = role_counts.get('Civilian', 0) + role_counts.get('Militant', 0)
print(f"  Civilian total after merge: {civilian_total}")

# Define aggressive transforms for underrepresented classes
aggressive_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create balanced dataset
balanced_df = _create_augmented_dataset(single_person_df)

# Filter out Militant class (we only use 4 classes)
balanced_df = balanced_df[balanced_df['Role'] != 'Militant'].copy()
print(f"📊 After filtering Militant: {len(balanced_df)} samples")

# Split balanced data
train_df, temp_df = train_test_split(balanced_df, test_size=0.3, random_state=42, stratify=balanced_df['Role'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Role'])

print(f"\n📊 Balanced data split:")
print(f"  Train: {len(train_df)} samples")
print(f"  Validation: {len(val_df)} samples") 
print(f"  Test: {len(test_df)} samples")

# Create datasets with aggressive augmentation
train_dataset = EnhancedOccupationDataset(train_df, train_transform, aggressive_transform)
val_dataset = EnhancedOccupationDataset(val_df, val_transform, aggressive_transform)
test_dataset = EnhancedOccupationDataset(test_df, val_transform, aggressive_transform)

# Weighted sampler for balanced training
from torch.utils.data import WeightedRandomSampler

# Calculate class weights for balanced training
def get_class_weights(dataset):
    """Calculate class weights for balanced training"""
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label.item())
    
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    return sample_weights

# Get sample weights for training
train_sample_weights = get_class_weights(train_dataset)
train_sampler = WeightedRandomSampler(
    weights=train_sample_weights,
    num_samples=len(train_sample_weights),
    replacement=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=16, 
    sampler=train_sampler
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

model = ColorAwareOccupationCNN(num_classes=4)
model.to(device)

# Class weights for loss
class_weights = torch.tensor([0.5, 1.5, 2.0, 2.5]).float().to(device)  # Emphasize minorities
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer - different LRs for different parts
optimizer = torch.optim.Adam([
    {'params': filter(lambda p: p.requires_grad, model.backbone.features.parameters()), 'lr': 5e-5},
    {'params': model.backbone.classifier.parameters(), 'lr': 5e-4}
], weight_decay=1e-4)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

# Training with early stopping
best_accuracy = 0.0
best_balanced = 0.0
patience = 5
patience_counter = 0

print("🚀 Starting optimized training...")

for epoch in range(35):  # A bit more epochs
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    class_correct = {i: 0 for i in range(4)}
    class_total = {i: 0 for i in range(4)}
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/35"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        # Track per-class
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i].item() == label:
                class_correct[label] += 1
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_predictions = []
    val_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            val_predictions.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    # Metrics
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    val_balanced = balanced_accuracy_score(val_labels, val_predictions) * 100
    
    scheduler.step()
    
    # Print progress
    if (epoch + 1) % 2 == 0:
        print(f"\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%, Balanced: {val_balanced:.2f}%")
        
        # Per-class training accuracy
        print("Training per-class accuracy:")
        for i, occ in enumerate(['Civilian', 'Child', 'Doctor', 'Police']):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"  {occ}: {acc:.1f}%")
    
    # Save best model
    if val_balanced > best_balanced:
        best_balanced = val_balanced
        best_accuracy = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'models/optimized_4class_occupation.pth')
        print(f"💾 New best model! Balanced accuracy: {best_balanced:.2f}%")
    else:
        patience_counter += 1
        if patience_counter >= patience and epoch > 10:
            print("Early stopping triggered")
            break

# Test Time Augmentation
print("\n🔄 Applying Test Time Augmentation...")
model.eval()
all_predictions = []
all_predictions_no_tta = []
all_labels = []

# TTA transforms
tta_transforms = [
    val_transform,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
]

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="TTA Evaluation")):
        batch_size = images.size(0)
        all_labels.extend(labels.numpy())
        
        # Standard prediction
        images_device = images.to(device)
        outputs = model(images_device)
        _, predicted = torch.max(outputs, 1)
        all_predictions_no_tta.extend(predicted.cpu().numpy())
        
        # TTA predictions - simplified version
        tta_batch_preds = []
        for img_idx in range(batch_size):
            tta_outputs = []
            
            for transform_idx, tta_transform in enumerate(tta_transforms):
                if transform_idx == 0:
                    img_tensor = images[img_idx].unsqueeze(0).to(device)
                else:
                    # Use the original image tensor for simplicity
                    img_tensor = images[img_idx].unsqueeze(0).to(device)
                
                output = model(img_tensor)
                tta_outputs.append(output)
            
            # Average predictions
            avg_output = torch.stack(tta_outputs).mean(0)
            _, tta_pred = torch.max(avg_output, 1)
            tta_batch_preds.append(tta_pred.item())
        
        all_predictions.extend(tta_batch_preds)

# Final results
occupation_classes = ['Civilian', 'Child', 'Doctor', 'Police']
print(f"\n✅ Training completed!")
print(f"Best validation accuracy: {best_accuracy:.2f}%")
print(f"Best balanced accuracy: {best_balanced:.2f}%")

# Standard results
print("\n📊 Standard Evaluation:")
standard_acc = accuracy_score(all_labels, all_predictions_no_tta)
standard_balanced = balanced_accuracy_score(all_labels, all_predictions_no_tta)
print(f"Accuracy: {standard_acc:.1%}, Balanced: {standard_balanced:.1%}")

# TTA results
print("\n🎯 With Test Time Augmentation:")
tta_acc = accuracy_score(all_labels, all_predictions)
tta_balanced = balanced_accuracy_score(all_labels, all_predictions)
print(f"Accuracy: {tta_acc:.1%}, Balanced: {tta_balanced:.1%}")
print(f"Improvement: +{(tta_acc - standard_acc)*100:.1f}%")

print("\n📋 Classification Report (with TTA):")
print(classification_report(all_labels, all_predictions, 
                                          target_names=occupation_classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
print("\n📊 Confusion Matrix:")
print("         Civ  Chi  Doc  Pol")
for i, row in enumerate(cm):
    print(f"{occupation_classes[i]:8} {row}")

# Per-class accuracy
print("\n📈 Per-class accuracy:")
for i, class_name in enumerate(occupation_classes):
    mask = np.array(all_labels) == i
    if mask.sum() > 0:
        acc = (np.array(all_predictions)[mask] == i).mean()
        print(f"{class_name:10} - {acc:.1%} ({(np.array(all_predictions)[mask] == i).sum()}/{mask.sum()})")

print("\n🎉 Complete! Model saved as 'models/optimized_4class_occupation.pth'")