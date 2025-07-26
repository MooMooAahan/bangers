#!/usr/bin/env python3
"""
Real World Test - Modified Dataset
Test status model on actual 200 images from modified_dataset
This is the TRUE test of generalization performance
"""

import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Add parent directory to path
sys.path.append('../')

print("🌍 REAL WORLD TEST - Modified Dataset")
print("📊 Testing on actual 200 images the model has NEVER seen")

# Transfer Learning Model (same architecture as training)
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

# Load the ORIGINAL metadata for modified dataset
print("📂 Loading original metadata for modified dataset...")
metadata_df = pd.read_csv('../data/metadata.csv')

# Filter for images that actually exist in modified_dataset
modified_dataset_path = '../data/modified_dataset'
available_images = []

if os.path.exists(modified_dataset_path):
    for filename in os.listdir(modified_dataset_path):
        if filename.endswith('.png'):
            # Find corresponding metadata
            img_path = f"modified_dataset/{filename}"
            matching_rows = metadata_df[metadata_df['Filename'] == img_path]
            if len(matching_rows) > 0:
                available_images.append({
                    'filename': filename,
                    'full_path': img_path,
                    'metadata': matching_rows.iloc[0]
                })

print(f"📊 Found {len(available_images)} test images in modified_dataset")

if len(available_images) == 0:
    print("❌ No images found in modified_dataset!")
    print("🔧 Make sure the modified_dataset folder exists and contains .png files")
    exit(1)

# Create status labels for test images
def create_status_label(row):
    class_val = row.get('Class', 'Default')
    injured_val = row.get('Injured', 'False')
    
    if str(class_val).lower() == 'zombie':
        return 'corpse' if str(injured_val).lower() == 'true' else 'zombie'
    else:
        return 'injured' if str(injured_val).lower() == 'true' else 'healthy'

status_classes = ['healthy', 'injured', 'zombie', 'corpse']
test_data = []

for img_info in available_images:
    metadata_row = img_info['metadata']
    status = create_status_label(metadata_row)
    test_data.append({
        'filename': img_info['filename'],
        'full_path': img_info['full_path'],
        'status': status,
        'humanoid_count': metadata_row.get('HumanoidCount', 1),
        'role': metadata_row.get('Role', 'civilian')
    })

# Convert to DataFrame for analysis
test_df = pd.DataFrame(test_data)
print(f"\n📊 Test Dataset Breakdown:")
print(f"   Total images: {len(test_df)}")
print(f"   Status distribution:")
status_counts = test_df['status'].value_counts()
for status, count in status_counts.items():
    print(f"     {status.capitalize()}: {count}")

print(f"   Humanoid count distribution:")
humanoid_counts = test_df['humanoid_count'].value_counts()
for count, freq in humanoid_counts.items():
    print(f"     {count} person(s): {freq}")

# Load trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TransferStatusCNN(num_classes=4)

try:
    model.load_state_dict(torch.load('../models/transfer_status_baseline.pth', map_location=device))
    model.to(device)
    model.eval()
    print("\n✅ Status model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("🔧 Make sure transfer_status_training.py has been run first")
    exit(1)

# Test transforms (same as training validation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create label encoder (same as training)
status_to_int = {status: i for i, status in enumerate(status_classes)}

print(f"\n🚀 Testing model on {len(test_df)} real-world images...")

# Test each image
predictions = []
true_labels = []
confidences = []
single_person_correct = 0
single_person_total = 0

for idx, row in test_df.iterrows():
    # Load and preprocess image
    img_path = os.path.join('../data', row['full_path'])
    
    try:
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get original size for info
        orig_size = image.size
        
        # Transform for model
        image_tensor = test_transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence_scores = probabilities.cpu().numpy()[0]
            predicted_idx = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            predicted_status = status_classes[predicted_idx]
            confidence = confidence_scores[predicted_idx] * 100
        
        # Record results
        true_status = row['status']
        true_idx = status_to_int[true_status]
        
        predictions.append(predicted_idx)
        true_labels.append(true_idx)
        confidences.append(confidence)
        
        # Track single-person accuracy separately
        if row['humanoid_count'] == 1:
            single_person_total += 1
            if predicted_idx == true_idx:
                single_person_correct += 1
        
        # Print detailed results for first few images
        if idx < 10:
            correct_mark = "✅" if predicted_idx == true_idx else "❌"
            print(f"   {correct_mark} {row['filename']}: True={true_status}, Pred={predicted_status} ({confidence:.1f}%) [Size: {orig_size}, People: {row['humanoid_count']}]")
    
    except Exception as e:
        print(f"   ❌ Error processing {row['filename']}: {e}")
        continue

# Calculate overall accuracy
overall_accuracy = (np.array(predictions) == np.array(true_labels)).mean() * 100
single_person_accuracy = (single_person_correct / single_person_total * 100) if single_person_total > 0 else 0

print(f"\n🎯 REAL WORLD PERFORMANCE RESULTS:")
print(f"📊 Overall Accuracy: {overall_accuracy:.1f}% ({sum(np.array(predictions) == np.array(true_labels))}/{len(predictions)})")
print(f"👤 Single-Person Accuracy: {single_person_accuracy:.1f}% ({single_person_correct}/{single_person_total})")

# Per-class accuracy
print(f"\n📊 Per-Class Performance:")
for i, class_name in enumerate(status_classes):
    class_mask = np.array(true_labels) == i
    if class_mask.sum() > 0:
        class_correct = (np.array(predictions)[class_mask] == i).sum()
        class_total = class_mask.sum()
        class_acc = (class_correct / class_total) * 100
        print(f"   {class_name.capitalize()}: {class_acc:.1f}% ({class_correct}/{class_total})")

# Create visualization
plt.figure(figsize=(12, 8))

# Confusion Matrix
plt.subplot(2, 2, 1)
cm = sklearn.metrics.confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=status_classes, yticklabels=status_classes,
            cbar=False)
plt.title('🌍 Real World Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Status')
plt.xlabel('Predicted Status')

# Per-class accuracy bars
plt.subplot(2, 2, 2)
per_class_acc = []
for i, class_name in enumerate(status_classes):
    class_mask = np.array(true_labels) == i
    if class_mask.sum() > 0:
        class_acc = (np.array(predictions)[class_mask] == i).mean() * 100
        per_class_acc.append(class_acc)
    else:
        per_class_acc.append(0)

colors = ['#2ecc71' if acc >= 90 else '#3498db' if acc >= 70 else '#f39c12' if acc >= 50 else '#e74c3c' for acc in per_class_acc]
bars = plt.bar(status_classes, per_class_acc, color=colors)
plt.title('📊 Real World Per-Class Accuracy', fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
for bar, acc in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

# Confidence distribution
plt.subplot(2, 2, 3)
plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('🎯 Prediction Confidence Distribution', fontweight='bold')
plt.xlabel('Confidence (%)')
plt.ylabel('Frequency')
plt.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.1f}%')
plt.legend()

# Summary statistics
plt.subplot(2, 2, 4)
summary_text = f"""REAL WORLD TEST SUMMARY

📊 Dataset: {len(test_df)} images from modified_dataset
🎯 Overall Accuracy: {overall_accuracy:.1f}%
👤 Single-Person Accuracy: {single_person_accuracy:.1f}%
📈 Mean Confidence: {np.mean(confidences):.1f}%

Status Distribution:
"""
for status, count in status_counts.items():
    summary_text += f"• {status.capitalize()}: {count}\n"

plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
plt.axis('off')

plt.tight_layout()
plt.savefig('real_world_test_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Classification report
if len(set(true_labels)) > 1:  # Only if we have multiple classes
    print(f"\n📋 Detailed Classification Report:")
    report = sklearn.metrics.classification_report(
        true_labels, predictions, 
        target_names=status_classes,
        zero_division=0
    )
    print(report)

# Find worst performing images
errors = []
for i, (true_idx, pred_idx, conf, row) in enumerate(zip(true_labels, predictions, confidences, test_df.itertuples())):
    if true_idx != pred_idx:
        true_class = status_classes[true_idx]
        pred_class = status_classes[pred_idx]
        errors.append({
            'filename': row.filename,
            'true': true_class,
            'pred': pred_class,
            'confidence': conf,
            'humanoid_count': row.humanoid_count
        })

if errors:
    print(f"\n⚠️ ERRORS FOUND ({len(errors)}/{len(predictions)} = {len(errors)/len(predictions)*100:.1f}%):")
    # Sort by confidence (high confidence errors are most concerning)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    for i, error in enumerate(errors[:10]):  # Show top 10 errors
        print(f"   {i+1}. {error['filename']}: {error['true']} → {error['pred']} ({error['confidence']:.1f}% conf, {error['humanoid_count']} people)")
    if len(errors) > 10:
        print(f"   ... and {len(errors)-10} more errors")
else:
    print(f"\n🎉 PERFECT PERFORMANCE - NO ERRORS FOUND!")

print(f"\n📊 Results saved as: real_world_test_results.png")
print(f"✅ Real world testing complete!")

# Final verdict
if overall_accuracy >= 90:
    print(f"\n🎉 EXCELLENT! {overall_accuracy:.1f}% accuracy on unseen data!")
    print("✅ Model generalizes very well to real-world images")
elif overall_accuracy >= 70:
    print(f"\n✅ GOOD! {overall_accuracy:.1f}% accuracy meets the 70% target")
    print("📈 Ready for production use")
elif overall_accuracy >= 50:
    print(f"\n⚠️ MODERATE: {overall_accuracy:.1f}% accuracy - needs improvement")
    print("🔧 Consider more training data or different approach")
else:
    print(f"\n❌ POOR: {overall_accuracy:.1f}% accuracy - major issues")
    print("🔧 Model needs significant improvements") 