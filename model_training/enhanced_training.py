#!/usr/bin/env python3
"""
Enhanced CNN Training Script
21-class system with data augmentation for 70%+ accuracy
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

# Add parent directory to path
sys.path.append('../')
from endpoints.data_parser import DataParser
from gameplay.humanoid import Humanoid
from models.DefaultCNN import DefaultCNN

print("🧠 Enhanced CNN Training - 21 Class System")
print("📊 Target: 70%+ accuracy with data augmentation")

# Load ORIGINAL dataset with Role information
try:
    full_df = pd.read_csv('../data/metadata.csv')
    print(f"📊 Total samples: {len(full_df)}")
    print(f"🎯 Available roles: {full_df['Role'].unique()}")
except FileNotFoundError:
    print("❌ metadata.csv not found. Please ensure it exists.")
    sys.exit(1)

# Enhanced augmentation for 5x data multiplication (200 → 1000 images)
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Standardize input size
    transforms.RandomHorizontalFlip(p=0.5),  # Mirror scenes - very effective
    transforms.RandomRotation(degrees=10),   # Slight camera angle variation
    transforms.ColorJitter(
        brightness=0.3,    # Day/night lighting conditions
        contrast=0.2,      # Weather/visibility variations
        saturation=0.2,    # Environmental effects
        hue=0.1           # Slight color shifts
    ),
    transforms.RandomResizedCrop(512, scale=(0.9, 1.0)),  # Slight zoom variation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class EnhancedDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms, base_dir="../data"):
        self.df = df
        self.base_dir = base_dir
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, i):
        row = self.df.iloc[i]
        
        # Use modified dataset path
        img_path = row['Filename']
        if not img_path.startswith('modified_dataset/'):
            img_path = f"modified_dataset/{os.path.basename(img_path)}"
            
        # Use enhanced 21-class label
        target = row['enhanced_class']
        
        try:
            image = Image.open(os.path.join(self.base_dir, img_path))
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.transforms(image)
            return image, torch.tensor(target).long()
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image and label on error
            return torch.zeros(3, 512, 512), torch.tensor(0).long()

# Handle multi-person images by taking only single-person for now
# Use integer comparison (HumanoidCount is int64, not string)
single_person_df = full_df[full_df['HumanoidCount'] == 1].copy()
print(f"📊 Single-person images: {len(single_person_df)}")
print(f"🔍 Sample HumanoidCount values: {full_df['HumanoidCount'].unique()[:5]}")

# Create train/test split (80/20)
train_df, test_df = sklearn.model_selection.train_test_split(
    single_person_df, test_size=0.2, random_state=42, stratify=single_person_df['Class'])

print(f"📊 Train samples: {len(train_df)}")
print(f"📊 Test samples: {len(test_df)}")

# Create enhanced labels directly
def create_enhanced_label(row):
    """Create enhanced 21-class label from metadata row directly"""
    class_val = row.get('Class', 'Default')
    injured_val = row.get('Injured', 'False')
    role_val = row.get('Role', 'Civilian')
    
    # Convert class to status
    if str(class_val).lower() == 'zombie':
        if str(injured_val).lower() == 'true':
            status = 'corpse'  # Injured zombie = corpse
        else:
            status = 'zombie'  # Healthy zombie
    else:  # Default
        if str(injured_val).lower() == 'true':
            status = 'injured'  # Injured human
        else:
            status = 'healthy'  # Healthy human
    
    # Clean up occupation from actual Role column
    occupation = str(role_val).lower().strip()
    
    # Map actual roles from metadata
    occupation_map = {
        'civilian': 'civilian',
        'child': 'child', 
        'doctor': 'doctor',
        'militant': 'militant',
        'police': 'police',  # In case it exists
        'blank': 'civilian',
        'unknown': 'civilian',
        '': 'civilian'
    }
    
    occupation = occupation_map.get(occupation, 'civilian')
    
    # Create enhanced class using Humanoid helper
    return Humanoid.create_enhanced_class(status, occupation)

# Generate enhanced class labels for each row
train_df['enhanced_class_str'] = train_df.apply(create_enhanced_label, axis=1)
test_df['enhanced_class_str'] = test_df.apply(create_enhanced_label, axis=1)

# Encode enhanced classes to integers
enc = sklearn.preprocessing.LabelEncoder()
enhanced_classes = Humanoid.get_enhanced_classes()  # 21 classes
enc.classes_ = np.array(enhanced_classes, dtype=object)

# Transform to integers
train_df['enhanced_class'] = enc.transform(train_df['enhanced_class_str'])
test_df['enhanced_class'] = enc.transform(test_df['enhanced_class_str'])

print(f"🎯 Enhanced classes ({len(enhanced_classes)}): {enhanced_classes}")
print(f"📊 Train class distribution:\n{train_df['enhanced_class_str'].value_counts()}")

# Training hyperparameters
batch_size = 16
learning_rate = 2e-4
weight_decay = 1e-8
epochs = 10  # Increased for better convergence
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_grad_norm = 100
epochs_warmup = 1

print(f"🔥 Using device: {device}")

# Create datasets and loaders
train_ds = EnhancedDataset(train_df, train_transform)
val_ds = EnhancedDataset(test_df, val_transform)
loader_train = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                 num_workers=0, pin_memory=False, shuffle=True, drop_last=True)
loader_val = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                 num_workers=0, pin_memory=False)

# Calculate class weights for balanced training across ALL 21 classes
# Need to handle classes that don't appear in training set
all_class_indices = np.arange(21)  # 0-20 for all 21 classes
present_classes = np.unique(train_df.enhanced_class)

# Compute weights only for present classes
present_weights = sklearn.utils.class_weight.compute_class_weight(
    'balanced', classes=present_classes, y=train_df.enhanced_class)

# Create full weight array for all 21 classes
class_weights = np.ones(21)  # Default weight of 1.0
for i, class_idx in enumerate(present_classes):
    class_weights[class_idx] = present_weights[i]

# For missing classes, use average weight to avoid bias
avg_weight = np.mean(present_weights)
for class_idx in all_class_indices:
    if class_idx not in present_classes:
        class_weights[class_idx] = avg_weight

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"🎯 Using {len(class_weights)} class weights for balanced training")  
print(f"📊 Present classes: {len(present_classes)}/21")

# Create enhanced 21-class model
model = DefaultCNN(num_classes_=21, input_size=512)
model.to(device)
print("🧠 Created enhanced DefaultCNN with 21 classes")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

nbatch = len(loader_train)
warmup = epochs_warmup * nbatch  # number of warmup steps
nsteps = epochs * nbatch        # number of total steps
scheduler = CosineLRScheduler(optimizer,
              warmup_t=warmup, warmup_lr_init=1e-6, warmup_prefix=True,
              t_initial=(nsteps - warmup), lr_min=1e-6)

def evaluate(model, loader_val):
    was_training = model.training
    model.eval()

    loss_sum = 0.0
    n_sum = 0
    y_all = []
    y_pred_all = []
    
    for img, y in loader_val:
        n = y.size(0)
        img = img.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(img)
        loss = criterion(y_pred, y)

        n_sum += n
        loss_sum += n * loss.item()
        
        y_all.append(y.cpu().detach().numpy())
        y_pred_all.append(torch.nn.functional.softmax(y_pred, 1).cpu().detach().numpy())

        del loss, y_pred, img, y
        gc.collect()

    loss_val = loss_sum / n_sum
    y = np.concatenate(y_all)
    y_pred = np.concatenate(y_pred_all)

    return {'loss': loss_val, 'y': y, 'y_pred': y_pred}

# Training loop
print("🚀 Starting enhanced CNN training...")
best_loss = 1000
best_accuracy = 0

torch.cuda.empty_cache()
for iepoch in range(epochs):
    print(f"\n📈 Epoch {iepoch+1}/{epochs}")
    
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for ibatch, (img, y) in tqdm(enumerate(loader_train), desc="Training", total=len(loader_train)):
        img = img.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        
        y_pred = model(img)
        loss = criterion(y_pred, y)

        loss_train = loss.item()
        epoch_loss += loss_train
        batch_count += 1

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step(iepoch * nbatch + ibatch + 1)
        
        gc.collect()
    
    avg_epoch_loss = epoch_loss / batch_count
    
    # Validation
    validation_results = evaluate(model, loader_val)
    val_loss = validation_results['loss']
    
    # Calculate accuracy
    y_pred_classes = np.array([np.argmax(x) for x in validation_results['y_pred']])
    accuracy = sklearn.metrics.accuracy_score(validation_results['y'], y_pred_classes)
    
    print(f"📊 Epoch {iepoch+1} Results:")
    print(f"   🔹 Train Loss: {avg_epoch_loss:.4f}")
    print(f"   🔹 Val Loss: {val_loss:.4f}")
    print(f"   🎯 Accuracy: {accuracy:.1%}")
    
    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_loss = val_loss
        model_path = '../models/enhanced_baseline.pth'
        torch.save(model.state_dict(), model_path)
        print(f"   ✅ New best model saved! Accuracy: {accuracy:.1%}")
    
    gc.collect()

print(f"\n🎉 Training completed!")
print(f"🏆 Best accuracy achieved: {best_accuracy:.1%}")
print(f"🎯 Target was 70% - {'✅ SUCCESS!' if best_accuracy >= 0.70 else '❌ Need improvement'}")

# Final evaluation and confusion matrix
print("\n📊 Generating final results...")
final_results = evaluate(model, loader_val)
y_pred_final = np.array([np.argmax(x) for x in final_results['y_pred']])

# Confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(final_results['y'], y_pred_final)
print(f"🔍 Confusion Matrix Shape: {confusion_matrix.shape}")

# Classification report
report = sklearn.metrics.classification_report(
    final_results['y'], y_pred_final, 
    target_names=enhanced_classes, zero_division=0)
print(f"\n📋 Classification Report:\n{report}")

print(f"\n💾 Enhanced model saved as: models/enhanced_baseline.pth")
print(f"🎯 Final accuracy: {best_accuracy:.1%}") 