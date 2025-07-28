#!/usr/bin/env python3
"""
Comprehensive Test for Enhanced Occupation CNN
Tests the model on the entire dataset to get complete performance metrics
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print("🧪 Comprehensive Occupation CNN Test")
print("=" * 50)

# Load metadata
try:
    full_df = pd.read_csv('data/metadata.csv')
    print(f"📊 Total samples in dataset: {len(full_df)}")
except FileNotFoundError:
    print("❌ metadata.csv not found")
    sys.exit(1)

# Define the model architecture (same as training)
class ColorAwareOccupationCNN(nn.Module):
    """Enhanced model with color feature extraction"""
    def __init__(self, num_classes=4):
        super(ColorAwareOccupationCNN, self).__init__()
        
        # MobileNetV2 - worked well before
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
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

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

model = ColorAwareOccupationCNN(num_classes=4)
try:
    model.load_state_dict(torch.load('models/optimized_4class_occupation.pth', map_location=device))
    print("✅ Enhanced occupation model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

model.to(device)
model.eval()

# Transform for testing
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Filter data for testing
print("\n📊 Filtering dataset for testing...")
single_person_df = full_df[full_df['HumanoidCount'] == 1].copy()
single_person_df = single_person_df[~single_person_df['Role'].str.contains('Beaver', case=False, na=False)]
valid_occupations = ['Civilian', 'Child', 'Doctor', 'Police', 'Militant']
single_person_df = single_person_df[single_person_df['Role'].isin(valid_occupations)]

print(f"📊 Single-person samples: {len(single_person_df)}")

# Show class distribution
role_counts = single_person_df['Role'].value_counts()
print("\n📈 Class distribution:")
for occ, count in role_counts.items():
    if occ == 'Militant':
        print(f"  {occ}: {count} (→ will merge with Civilian)")
    else:
        print(f"  {occ}: {count}")

# Occupation classes
occupation_classes = ['Civilian', 'Child', 'Doctor', 'Police']

# Test the model
print("\n🧪 Testing model on entire dataset...")
all_predictions = []
all_labels = []
all_filenames = []
failed_images = []

with torch.no_grad():
    for idx, row in tqdm(single_person_df.iterrows(), total=len(single_person_df), desc="Testing"):
        try:
            # Load image
            img_path = os.path.join('data', row['Filename'])
            image = Image.open(img_path).convert('RGB')
            
            # Transform
            image_tensor = test_transform(image).unsqueeze(0).to(device)
            
            # Get prediction
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()
            
            # Get true label
            role = row['Role']
            if role == 'Militant':
                role = 'Civilian'  # Merge Militant with Civilian
            true_label = occupation_classes.index(role)
            
            # Store results
            all_predictions.append(prediction)
            all_labels.append(true_label)
            all_filenames.append(row['Filename'])
            
        except Exception as e:
            failed_images.append((row['Filename'], str(e)))
            print(f"❌ Failed to process {row['Filename']}: {e}")

print(f"\n✅ Testing completed!")
print(f"📊 Successfully processed: {len(all_predictions)} images")
print(f"❌ Failed to process: {len(failed_images)} images")

if failed_images:
    print("\n📋 Failed images:")
    for filename, error in failed_images[:5]:  # Show first 5
        print(f"  {filename}: {error}")

# Calculate metrics
print("\n📊 Performance Metrics:")
print("=" * 50)

# Overall accuracy
overall_acc = accuracy_score(all_labels, all_predictions)
balanced_acc = balanced_accuracy_score(all_labels, all_predictions)

print(f"🎯 Overall Accuracy: {overall_acc:.1%}")
print(f"⚖️  Balanced Accuracy: {balanced_acc:.1%}")

# Per-class accuracy
print("\n📈 Per-class Performance:")
class_correct = {i: 0 for i in range(4)}
class_total = {i: 0 for i in range(4)}

for true_label, pred_label in zip(all_labels, all_predictions):
    class_total[true_label] += 1
    if true_label == pred_label:
        class_correct[true_label] += 1

for i, class_name in enumerate(occupation_classes):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i]
        print(f"  {class_name:10} - {acc:.1%} ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"  {class_name:10} - No samples")

# Classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(all_labels, all_predictions, 
                          target_names=occupation_classes,
                          digits=3))

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
print("\n📊 Confusion Matrix:")
print("         Civ  Chi  Doc  Pol")
for i, row in enumerate(cm):
    print(f"{occupation_classes[i]:8} {row}")

# Create confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=occupation_classes, 
            yticklabels=occupation_classes)
plt.title('Occupation CNN Confusion Matrix\n(Enhanced Model on Full Dataset)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('enhanced_occupation_full_test_confusion.png', dpi=300, bbox_inches='tight')
print("\n💾 Confusion matrix saved as 'enhanced_occupation_full_test_confusion.png'")

# Error analysis
print("\n🔍 Error Analysis:")
print("=" * 50)

# Find most common errors
error_pairs = []
for i, (true_label, pred_label) in enumerate(zip(all_labels, all_predictions)):
    if true_label != pred_label:
        error_pairs.append((occupation_classes[true_label], occupation_classes[pred_label], all_filenames[i]))

if error_pairs:
    print(f"📊 Total errors: {len(error_pairs)}")
    
    # Count error types
    error_counts = {}
    for true_class, pred_class, _ in error_pairs:
        error_key = f"{true_class} → {pred_class}"
        error_counts[error_key] = error_counts.get(error_key, 0) + 1
    
    print("\n📈 Most common errors:")
    for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {error_type}: {count} times")
    
    # Show some example errors
    print("\n📋 Example error cases:")
    for i, (true_class, pred_class, filename) in enumerate(error_pairs[:5]):
        print(f"  {filename}: {true_class} → {pred_class}")

# Summary
print("\n🎉 Test Summary:")
print("=" * 50)
print(f"✅ Model: Enhanced Occupation CNN (optimized_4class_occupation.pth)")
print(f"📊 Dataset: {len(all_predictions)} single-person images")
print(f"🎯 Overall Accuracy: {overall_acc:.1%}")
print(f"⚖️  Balanced Accuracy: {balanced_acc:.1%}")
print(f"📈 Best performing class: {occupation_classes[np.argmax([class_correct[i]/class_total[i] if class_total[i] > 0 else 0 for i in range(4)])]}")
print(f"📉 Worst performing class: {occupation_classes[np.argmin([class_correct[i]/class_total[i] if class_total[i] > 0 else 1 for i in range(4)])]}")

print("\n🎯 Test completed successfully!") 