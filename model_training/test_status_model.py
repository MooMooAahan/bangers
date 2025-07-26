#!/usr/bin/env python3
"""
Test Status Model Performance
Comprehensive testing with visual results and sample predictions
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Add parent directory to path
sys.path.append('../')

print("🧪 Testing Status Classification Model")
print("📊 Loading model and testing comprehensive performance")

# Transfer Learning Model (same as training)
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

# Load test data
full_df = pd.read_csv('../data/metadata.csv')
single_person_df = full_df[full_df['HumanoidCount'] == 1].copy()

def create_status_label(row):
    class_val = row.get('Class', 'Default')
    injured_val = row.get('Injured', 'False')
    
    if str(class_val).lower() == 'zombie':
        return 'corpse' if str(injured_val).lower() == 'true' else 'zombie'
    else:
        return 'injured' if str(injured_val).lower() == 'true' else 'healthy'

single_person_df['status'] = single_person_df.apply(create_status_label, axis=1)

# Balance and encode (same as training)
status_counts = single_person_df['status'].value_counts()
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
    else:
        balanced_status_df = status_df
    
    balanced_dfs.append(balanced_status_df)

balanced_df = pd.concat(balanced_dfs, ignore_index=True)

# Encode status
status_classes = ['healthy', 'injured', 'zombie', 'corpse']
status_encoder = sklearn.preprocessing.LabelEncoder()
status_encoder.classes_ = np.array(status_classes, dtype=object)
balanced_df['status_encoded'] = status_encoder.transform(balanced_df['status'])

# Same train/test split as training
train_df, test_df = sklearn.model_selection.train_test_split(
    balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['status'])

print(f"📊 Test set size: {len(test_df)} images")
print(f"📊 Test distribution: {test_df['status'].value_counts().to_dict()}")

# Load trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TransferStatusCNN(num_classes=4)

try:
    model.load_state_dict(torch.load('../models/transfer_status_baseline.pth', map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("🔧 Make sure transfer_status_training.py has been run first")
    exit(1)

# Test transforms (same as validation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TestDataset(torch.utils.data.Dataset):
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
        
        image = Image.open(os.path.join(self.base_dir, img_path))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transforms(image)
        return image, torch.tensor(status_label).long(), img_path

# Test the model
test_ds = TestDataset(test_df, test_transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

print(f"\n🚀 Testing model on {len(test_ds)} images...")

all_preds = []
all_labels = []
all_probs = []
all_paths = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels, paths in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())
        all_paths.extend(paths)

test_accuracy = 100 * correct / total
print(f"🎯 Test Accuracy: {test_accuracy:.1f}%")

# Create confusion matrix visualization
plt.figure(figsize=(10, 8))
cm = sklearn.metrics.confusion_matrix(all_labels, all_preds)

# Create the "4 squares" visualization
plt.subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=status_classes, yticklabels=status_classes,
            cbar=False)
plt.title('🔍 Confusion Matrix\n"The 4 Squares"', fontsize=14, fontweight='bold')
plt.ylabel('True Status')
plt.xlabel('Predicted Status')

# Detailed per-class results
plt.subplot(2, 2, 2)
per_class_acc = []
for i, class_name in enumerate(status_classes):
    class_mask = np.array(all_labels) == i
    if class_mask.sum() > 0:
        class_acc = (np.array(all_preds)[class_mask] == i).mean() * 100
        per_class_acc.append(class_acc)
    else:
        per_class_acc.append(0)

colors = ['#2ecc71' if acc == 100 else '#3498db' if acc >= 80 else '#f39c12' if acc >= 60 else '#e74c3c' for acc in per_class_acc]
bars = plt.bar(status_classes, per_class_acc, color=colors)
plt.title('📊 Per-Class Accuracy', fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
for bar, acc in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

# Sample predictions with confidence
plt.subplot(2, 1, 2)
sample_results = []
for i in range(min(10, len(all_preds))):
    true_class = status_classes[all_labels[i]]
    pred_class = status_classes[all_preds[i]]
    confidence = all_probs[i][all_preds[i]] * 100
    correct_mark = "✅" if all_labels[i] == all_preds[i] else "❌"
    sample_results.append(f"{correct_mark} True: {true_class} | Pred: {pred_class} ({confidence:.1f}%)")

plt.text(0.05, 0.95, '\n'.join(sample_results), transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
plt.title('🎯 Sample Predictions with Confidence', fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('status_model_test_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Detailed classification report
print(f"\n📋 Detailed Classification Report:")
report = sklearn.metrics.classification_report(
    all_labels, all_preds, 
    target_names=status_classes,
    zero_division=0
)
print(report)

# Performance summary
print(f"\n🎉 PERFORMANCE SUMMARY:")
print(f"🎯 Overall Accuracy: {test_accuracy:.1f}%")
print(f"📊 Per-Class Performance:")
for i, class_name in enumerate(status_classes):
    class_mask = np.array(all_labels) == i
    if class_mask.sum() > 0:
        class_acc = (np.array(all_preds)[class_mask] == i).mean() * 100
        print(f"   {class_name.capitalize()}: {class_acc:.1f}%")

# Find any errors for analysis
errors = []
for i, (true_label, pred_label, prob, path) in enumerate(zip(all_labels, all_preds, all_probs, all_paths)):
    if true_label != pred_label:
        true_class = status_classes[true_label]
        pred_class = status_classes[pred_label]
        confidence = prob[pred_label] * 100
        errors.append(f"❌ {path}: True={true_class}, Pred={pred_class} ({confidence:.1f}%)")

if errors:
    print(f"\n⚠️ ERRORS FOUND ({len(errors)} total):")
    for error in errors[:5]:  # Show first 5 errors
        print(f"   {error}")
    if len(errors) > 5:
        print(f"   ... and {len(errors)-5} more")
else:
    print(f"\n🎉 PERFECT PERFORMANCE - NO ERRORS FOUND!")

print(f"\n📊 Confusion matrix saved as: status_model_test_results.png")
print(f"✅ Status classification model testing complete!") 