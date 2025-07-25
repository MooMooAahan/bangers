#!/usr/bin/env python3
"""
🧠 CNN Accuracy Testing Suite
Tests CNN accuracy against ground truth data before replacing 100% accurate inspect feature
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append('.')

from endpoints.heuristic_interface import Predictor
from endpoints.data_parser import DataParser
from gameplay.humanoid import Humanoid
from gameplay.enums import State


class CNNAccuracyTester:
    """
    Comprehensive CNN accuracy testing and evaluation
    """
    
    def __init__(self, model_file='models/baseline.pth', data_root='data'):
        self.model_file = model_file
        self.data_root = data_root
        self.predictor = Predictor(classes=4, model_file=model_file)
        self.data_parser = DataParser(data_root)
        
        # Track results
        self.predictions = []
        self.actuals = []
        self.confidences = []
        self.image_files = []
        self.occupations = []
        
    def test_single_image(self, image_path, actual_state, occupation=None, verbose=True):
        """Test CNN accuracy on a single image"""
        try:
            img = Image.open(image_path)
            probs = self.predictor.get_probs(img)
            
            predicted_idx = np.argmax(probs)
            predicted_state = Humanoid.get_all_states()[predicted_idx]
            confidence = probs[predicted_idx]
            
            is_correct = predicted_state == actual_state
            
            if verbose:
                print(f"🖼️  Image: {os.path.basename(image_path)}")
                print(f"   🎯 Predicted: {predicted_state} ({confidence:.3f})")
                print(f"   ✅ Actual: {actual_state}")
                print(f"   {'✅ CORRECT' if is_correct else '❌ WRONG'}")
                if occupation:
                    print(f"   👤 Occupation: {occupation}")
                print()
            
            return {
                'predicted': predicted_state,
                'actual': actual_state,
                'confidence': confidence,
                'correct': is_correct,
                'probs': probs
            }
            
        except Exception as e:
            print(f"❌ Error testing {image_path}: {e}")
            return None
    
    def test_from_metadata(self, max_samples=100, verbose_per_image=False):
        """Test CNN accuracy using metadata CSV"""
        print(f"🧪 Testing CNN Accuracy on {max_samples} samples...")
        print(f"📁 Model: {self.model_file}")
        print(f"📊 Data: {self.data_root}")
        print("=" * 80)
        
        # Reset results
        self.predictions = []
        self.actuals = []
        self.confidences = []
        self.image_files = []
        self.occupations = []
        
        # Sample from dataset
        sample_indices = np.random.choice(len(self.data_parser.df), 
                                        min(max_samples, len(self.data_parser.df)), 
                                        replace=False)
        
        correct_count = 0
        total_count = 0
        
        for idx in sample_indices:
            row = self.data_parser.df.iloc[idx]
            
            # Handle compound images (HumanoidCount=2)
            humanoid_count = int(row.get('HumanoidCount', 1))
            
            if humanoid_count == 1:
                # Single humanoid
                image_path = os.path.join(self.data_root, row['Filename'])
                actual_class = row['Class']
                actual_injured = row.get('Injured', 'False')
                
                # Convert to state
                if actual_class.lower() == 'zombie':
                    actual_state = 'corpse' if actual_injured == 'True' else 'zombie'
                else:  # Default
                    actual_state = 'injured' if actual_injured == 'True' else 'healthy'
                
                result = self.test_single_image(image_path, actual_state, 
                                              row.get('Role', 'Unknown'), 
                                              verbose_per_image)
                
                if result:
                    self.predictions.append(result['predicted'])
                    self.actuals.append(result['actual'])
                    self.confidences.append(result['confidence'])
                    self.image_files.append(row['Filename'])
                    self.occupations.append(row.get('Role', 'Unknown'))
                    
                    if result['correct']:
                        correct_count += 1
                    total_count += 1
                    
            elif humanoid_count == 2:
                # Compound image - test against both people
                image_path = os.path.join(self.data_root, row['Filename'])
                
                classes = row['Class'].split('|')
                injuries = row['Injured'].split('|')
                roles = row.get('Role', 'Unknown|Unknown').split('|')
                
                for i, (cls, inj, role) in enumerate(zip(classes, injuries, roles)):
                    if cls.lower() == 'zombie':
                        actual_state = 'corpse' if inj == 'True' else 'zombie' 
                    else:
                        actual_state = 'injured' if inj == 'True' else 'healthy'
                    
                    result = self.test_single_image(image_path, actual_state, role, verbose_per_image)
                    
                    if result:
                        self.predictions.append(result['predicted'])
                        self.actuals.append(result['actual'])
                        self.confidences.append(result['confidence'])
                        self.image_files.append(f"{row['Filename']}_person{i+1}")
                        self.occupations.append(role)
                        
                        if result['correct']:
                            correct_count += 1
                        total_count += 1
        
        # Calculate overall accuracy
        if total_count > 0:
            overall_accuracy = correct_count / total_count
            print(f"🎯 Overall Accuracy: {overall_accuracy:.3f} ({correct_count}/{total_count})")
        else:
            print("❌ No valid predictions made")
            
        return self.calculate_detailed_metrics()
    
    def calculate_detailed_metrics(self):
        """Calculate comprehensive accuracy metrics"""
        if not self.predictions:
            print("❌ No predictions to analyze")
            return None
            
        # Convert to numerical for sklearn
        state_to_idx = {state: i for i, state in enumerate(Humanoid.get_all_states())}
        y_pred = [state_to_idx[pred] for pred in self.predictions]
        y_true = [state_to_idx[actual] for actual in self.actuals]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        print("\n📊 DETAILED METRICS:")
        print(f"   🎯 Accuracy: {accuracy:.3f}")
        print(f"   🎯 Precision: {precision:.3f}")
        print(f"   🎯 Recall: {recall:.3f}") 
        print(f"   🎯 F1-Score: {f1:.3f}")
        
        # Per-class metrics
        class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=list(range(len(Humanoid.get_all_states())))
        )
        
        print("\n📈 PER-CLASS METRICS:")
        for i, state in enumerate(Humanoid.get_all_states()):
            if i < len(class_precision):
                print(f"   {state.upper()}:")
                print(f"      Precision: {class_precision[i]:.3f}")
                print(f"      Recall: {class_recall[i]:.3f}")
                print(f"      F1-Score: {class_f1[i]:.3f}")
                print(f"      Support: {class_support[i]}")
        
        # Confidence analysis
        avg_confidence = np.mean(self.confidences)
        print(f"\n🔍 Average Confidence: {avg_confidence:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, Humanoid.get_all_states())
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'avg_confidence': avg_confidence
        }
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('CNN Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            
            # Save plot
            plot_path = 'CNN_confusion_matrix.png'
            plt.savefig(plot_path)
            print(f"📊 Confusion matrix saved to: {plot_path}")
            plt.show()
        except Exception as e:
            print(f"⚠️  Could not generate plot: {e}")
    
    def test_inspect_replacement_readiness(self):
        """Test if CNN is ready to replace 100% accurate inspect"""
        print("🔬 INSPECT REPLACEMENT READINESS TEST")
        print("=" * 60)
        
        metrics = self.test_from_metadata(max_samples=200, verbose_per_image=False)
        
        if metrics:
            accuracy = metrics['accuracy']
            avg_confidence = metrics['avg_confidence']
            
            print(f"\n🎯 READINESS ASSESSMENT:")
            
            if accuracy >= 0.90:
                print(f"✅ Excellent accuracy ({accuracy:.3f}) - Ready for inspect replacement!")
            elif accuracy >= 0.80:
                print(f"⚠️  Good accuracy ({accuracy:.3f}) - Consider additional training")
            elif accuracy >= 0.70:
                print(f"⚠️  Moderate accuracy ({accuracy:.3f}) - Needs improvement")
            else:
                print(f"❌ Low accuracy ({accuracy:.3f}) - Not ready for inspect replacement")
            
            if avg_confidence >= 0.80:
                print(f"✅ High confidence ({avg_confidence:.3f}) - Model is decisive")
            elif avg_confidence >= 0.60:
                print(f"⚠️  Moderate confidence ({avg_confidence:.3f}) - Some uncertainty")
            else:
                print(f"❌ Low confidence ({avg_confidence:.3f}) - Model is uncertain")
                
            # Recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            if accuracy < 0.85 or avg_confidence < 0.70:
                print("   📚 Consider additional training data")
                print("   🔧 Fine-tune hyperparameters")
                print("   🎯 Focus on misclassified categories")
            else:
                print("   ✅ CNN appears ready for inspect replacement")
                print("   📊 Monitor performance in production")
                print("   🔄 Implement fallback mechanisms")


def main():
    """Run CNN accuracy tests"""
    print("🧠 CNN Accuracy Testing Suite")
    print("Testing CNN before replacing 100% accurate inspect feature\n")
    
    # Initialize tester
    tester = CNNAccuracyTester()
    
    # Test inspect replacement readiness
    tester.test_inspect_replacement_readiness()
    
    print("\n" + "="*80)
    print("🎯 Testing complete! Check console output and saved plots.")
    print("Use this data to decide if CNN is ready to replace inspect feature.")


if __name__ == "__main__":
    main() 