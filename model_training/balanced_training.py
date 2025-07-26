#!/usr/bin/env python3
"""
Balanced CNN Training Script
Smart data augmentation to balance occupation classes for 70%+ accuracy
"""

import pandas as pd
import os
import sys
import torch
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.utils
import sklearn.metrics
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import gc
from collections import Counter

# Add parent directory to path
sys.path.append('../')
from gameplay.humanoid import Humanoid
from models.DefaultCNN import DefaultCNN

print("🎯 Balanced CNN Training - Smart Occupation Augmentation")
print("📊 Target: 70%+ accuracy with balanced occupation classes")

# Load dataset
full_df = pd.read_csv('../data/metadata.csv')
single_person_df = full_df[full_df['HumanoidCount'] == 1].copy()

print(f"📊 Single-person images: {len(single_person_df)}")
print(f"📈 Raw occupation distribution:")
role_counts = single_person_df['Role'].value_counts()
print(role_counts)

# Create enhanced labels
def create_enhanced_label(row):
    class_val = row.get('Class', 'Default')
    injured_val = row.get('Injured', 'False')
    role_val = row.get('Role', 'Civilian')
    
    if str(class_val).lower() == 'zombie':
        status = 'corpse' if str(injured_val).lower() == 'true' else 'zombie'
    else:
        status = 'injured' if str(injured_val).lower() == 'true' else 'healthy'
    
    occupation = str(role_val).lower().strip()
    occupation_map = {
        'civilian': 'civilian', 'child': 'child', 'doctor': 'doctor',
        'militant': 'militant', 'police': 'police'
    }
    occupation = occupation_map.get(occupation, 'civilian')
    
    return Humanoid.create_enhanced_class(status, occupation)

single_person_df['enhanced_class_str'] = single_person_df.apply(create_enhanced_label, axis=1)

# SMART BALANCING STRATEGY: Augment minority occupations
print("\n🎯 SMART BALANCING STRATEGY")
print("="*50)

# Group by occupation for balancing
single_person_df['occupation'] = single_person_df['Role'].str.lower()
occupation_counts = single_person_df['occupation'].value_counts()
print(f"📊 Current occupation distribution:")
print(occupation_counts)

# Target: Balance all occupations to match civilian count
target_count = occupation_counts['civilian']  # 54 samples
print(f"\n🎯 Target count per occupation: {target_count}")

balanced_dfs = []
augmentation_report = []

for occupation in occupation_counts.index:
    current_count = occupation_counts[occupation]
    occupation_df = single_person_df[single_person_df['occupation'] == occupation].copy()
    
    if current_count < target_count:
        # Calculate how many times to replicate
        multiplier = target_count // current_count
        remainder = target_count % current_count
        
        # Replicate the data
        replicated_dfs = [occupation_df] * multiplier
        if remainder > 0:
            replicated_dfs.append(occupation_df.sample(n=remainder, random_state=42))
        
        balanced_occupation_df = pd.concat(replicated_dfs, ignore_index=True)
        
        augmentation_report.append({
            'occupation': occupation,
            'original': current_count,
            'augmented': len(balanced_occupation_df),
            'multiplier': f"{multiplier}x + {remainder}"
        })
    else:
        balanced_occupation_df = occupation_df
        augmentation_report.append({
            'occupation': occupation,
            'original': current_count,
            'augmented': len(balanced_occupation_df),
            'multiplier': "1x (no change)"
        })
    
    balanced_dfs.append(balanced_occupation_df)

# Combine all balanced data
balanced_df = pd.concat(balanced_dfs, ignore_index=True)

print(f"\n📈 AUGMENTATION REPORT:")
for report in augmentation_report:
    print(f"  {report['occupation'].capitalize()}: {report['original']} → {report['augmented']} ({report['multiplier']})")

print(f"\n✅ Total augmented dataset: {len(balanced_df)} images")
print(f"📊 Enhanced class distribution after balancing:")
balanced_class_counts = balanced_df['enhanced_class_str'].value_counts()
print(balanced_class_counts.head(10))

# Create train/test split from balanced data
train_df, test_df = sklearn.model_selection.train_test_split(
    balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['Class'])

print(f"\n📊 Balanced train samples: {len(train_df)}")
print(f"📊 Balanced test samples: {len(test_df)}")

# Enhanced augmentation transforms
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # Increased for more variation
    transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(512, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms, base_dir="../data"):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = row['Filename']
        target = row['enhanced_class']
        
        try:
            image = Image.open(os.path.join(self.base_dir, img_path))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.transforms(image)
            return image, torch.tensor(target).long()
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 512, 512), torch.tensor(0).long()

# Encode enhanced classes
enc = sklearn.preprocessing.LabelEncoder()
enhanced_classes = Humanoid.get_enhanced_classes()
enc.classes_ = np.array(enhanced_classes, dtype=object)

train_df['enhanced_class'] = enc.transform(train_df['enhanced_class_str'])
test_df['enhanced_class'] = enc.transform(test_df['enhanced_class_str'])

# Training setup
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 15  # More epochs for better convergence

train_ds = BalancedDataset(train_df, train_transform)
val_ds = BalancedDataset(test_df, val_transform)
loader_train = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
loader_val = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

# Balanced class weights
present_classes = np.unique(train_df.enhanced_class)
present_weights = sklearn.utils.class_weight.compute_class_weight(
    'balanced', classes=present_classes, y=train_df.enhanced_class)

class_weights = np.ones(21)
for i, class_idx in enumerate(present_classes):
    class_weights[class_idx] = present_weights[i]

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Model setup
model = DefaultCNN(num_classes_=21, input_size=512)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

def evaluate(model, loader_val):
    model.eval()
    loss_sum = 0.0
    n_sum = 0
    y_all = []
    y_pred_all = []
    
    with torch.no_grad():
        for img, y in loader_val:
            img, y = img.to(device), y.to(device)
            y_pred = model(img)
            loss = criterion(y_pred, y)
            
            n_sum += y.size(0)
            loss_sum += y.size(0) * loss.item()
            
            y_all.append(y.cpu().numpy())
            y_pred_all.append(torch.softmax(y_pred, 1).cpu().numpy())
    
    return {
        'loss': loss_sum / n_sum,
        'y': np.concatenate(y_all),
        'y_pred': np.concatenate(y_pred_all)
    }

# Training loop
print(f"\n🚀 Starting balanced training on {device}...")
best_accuracy = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for img, y in tqdm(loader_train, desc=f"Epoch {epoch+1}/{epochs}"):
        img, y = img.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(img)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Validation
    val_results = evaluate(model, loader_val)
    y_pred_classes = np.argmax(val_results['y_pred'], axis=1)
    accuracy = sklearn.metrics.accuracy_score(val_results['y'], y_pred_classes)
    
    print(f"📊 Epoch {epoch+1}: Loss={epoch_loss/len(loader_train):.4f}, Val_Acc={accuracy:.1%}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), '../models/balanced_baseline.pth')
        print(f"   ✅ New best model saved! Accuracy: {accuracy:.1%}")

print(f"\n🎉 Training completed!")
print(f"🏆 Best accuracy achieved: {best_accuracy:.1%}")
print(f"🎯 Target was 70% - {'✅ SUCCESS!' if best_accuracy >= 0.70 else '❌ Need improvement'}")

# Final evaluation with fixed confusion matrix
final_results = evaluate(model, loader_val)
y_pred_final = np.argmax(final_results['y_pred'], axis=1)

# Get only the classes that actually appear in test set
unique_test_classes = np.unique(final_results['y'])
test_class_names = [enhanced_classes[i] for i in unique_test_classes]

print(f"\n📊 Final Results:")
print(f"🔍 Test set has {len(unique_test_classes)} unique classes")

# Fixed classification report
report = sklearn.metrics.classification_report(
    final_results['y'], y_pred_final, 
    labels=unique_test_classes,
    target_names=test_class_names, 
    zero_division=0
)
print(f"\n📋 Classification Report:\n{report}")

print(f"\n💾 Balanced model saved as: models/balanced_baseline.pth")
print(f"🎯 Final accuracy: {best_accuracy:.1%}") 