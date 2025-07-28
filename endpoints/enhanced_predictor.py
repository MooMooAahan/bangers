"""
Enhanced Predictor Interface
Combines status and occupation CNN predictions for RL agent and user interface
Provides side-specific predictions and imperfect CNN functionality for users
"""

import torch
import numpy as np
from PIL import Image
import random
import os # Added for path handling

# Manual transforms to avoid torchvision hanging issue
class ManualTransforms:
    @staticmethod
    def resize_and_normalize(image, size=(512, 512)):
        """Manual resize and normalize to avoid torchvision import hang"""
        # Resize image
        image = image.resize(size, Image.LANCZOS)
        
        # Convert to tensor
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image_array = (image_array - mean) / std
        
        # Convert to tensor and add batch dimension (ensure float32)
        tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float().unsqueeze(0)
        
        return tensor

from models.DefaultCNN import DefaultCNN
from models.OccupationCNN import OccupationCNN
from models.TransferStatusCNN import TransferStatusCNN # Added TransferStatusCNN import
from gameplay.enums import State


class EnhancedPredictor:
    """
    Enhanced predictor that combines status and occupation CNNs
    Provides both accurate predictions for RL and imperfect predictions for users
    """
    
    def __init__(self, status_model_file='models/transfer_status_baseline.pth', 
                 occupation_model_file='models/optimized_4class_occupation.pth'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load status model (3-class: healthy, injured, zombie)
        self.status_model = TransferStatusCNN(num_classes=3)
        try:
            state_dict = torch.load(status_model_file, map_location=self.device)
            self.status_model.load_state_dict(state_dict)
            print(f"✅ Status model loaded from {status_model_file}")
        except Exception as e:
            print(f"❌ Failed to load status model: {e}")
            self.status_model = None
        
        self.status_model.to(self.device)
        self.status_model.eval()
        self.status_classes = ['healthy', 'injured', 'zombie']  # Only 3 classes
        
        # Occupation CNN (new)
        self.occupation_model = OccupationCNN(num_classes=4, input_size=512)  # 4 classes (Militant removed)
        try:
            # Load state dict with compatibility handling
            state_dict = torch.load(occupation_model_file, map_location=self.device)
            
            # Filter out incompatible keys (fc layers are dynamically created)
            model_keys = set(self.occupation_model.state_dict().keys())
            state_keys = set(state_dict.keys())
            
            # Remove fc layer keys if they exist in saved model but not in current model
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                 if k in model_keys or not k.startswith('fc.')}
            
            # Load compatible layers only
            self.occupation_model.load_state_dict(filtered_state_dict, strict=False)
            self.occupation_model.to(self.device)
            self.occupation_model.eval()
            self.occupation_model_available = True
            print(f"✅ Occupation model loaded from {occupation_model_file} (filtered incompatible keys)")
        except FileNotFoundError:
            print(f"⚠️ Occupation model not found at {occupation_model_file}, using random predictions")
            self.occupation_model_available = False
        except Exception as e:
            print(f"⚠️ Error loading occupation model: {e}, using random predictions")
            self.occupation_model_available = False
        
        # Class mappings
        self.occupation_classes = ['Civilian', 'Child', 'Doctor', 'Police']  # 4 classes (Militant removed)
        
        # Image preprocessing (manual to avoid torchvision import hang)
        self.manual_transforms = ManualTransforms()
    
    def preprocess_image(self, image_path):
        """Preprocess image for CNN input"""
        try:
            # Handle both full paths and filenames
            if not os.path.exists(image_path):
                # If it's just a filename, construct the full path
                if not image_path.startswith('data/'):
                    # Check if filename already contains modified_dataset
                    if 'modified_dataset' in image_path:
                        image_path = os.path.join('data', image_path)
                    else:
                        image_path = os.path.join('data', 'modified_dataset', image_path)
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.manual_transforms.resize_and_normalize(image).to(self.device)
            return image_tensor
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def predict_status(self, image_path, return_probabilities=False):
        """Predict status using status CNN"""
        if not self.status_model:
            # Return random prediction if model not available
            predicted_class = random.randint(0, len(self.status_classes) - 1)
            probabilities = np.random.dirichlet(np.ones(len(self.status_classes)))
            
            if return_probabilities:
                return self.status_classes[predicted_class], probabilities
            return self.status_classes[predicted_class]
        
        image_tensor = self.preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        with torch.no_grad():
            outputs = self.status_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        if return_probabilities:
            return self.status_classes[predicted_class], probabilities
        return self.status_classes[predicted_class]
    
    def predict_occupation(self, image_path, return_probabilities=False):
        """Predict occupation using occupation CNN"""
        if not self.occupation_model_available:
            # Return random occupation if model not available
            predicted_class = random.randint(0, len(self.occupation_classes) - 1)
            probabilities = np.random.dirichlet(np.ones(len(self.occupation_classes)))
            
            if return_probabilities:
                return self.occupation_classes[predicted_class], probabilities
            return self.occupation_classes[predicted_class]
        
        image_tensor = self.preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        with torch.no_grad():
            outputs = self.occupation_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        if return_probabilities:
            return self.occupation_classes[predicted_class], probabilities
        return self.occupation_classes[predicted_class]
    
    def predict_combined(self, image_path, return_probabilities=False):
        """Get both status and occupation predictions"""
        status_result = self.predict_status(image_path, return_probabilities)
        occupation_result = self.predict_occupation(image_path, return_probabilities)
        
        if return_probabilities:
            status_class, status_probs = status_result
            occupation_class, occupation_probs = occupation_result
            return {
                'status': status_class,
                'occupation': occupation_class,
                'status_probabilities': status_probs,
                'occupation_probabilities': occupation_probs
            }
        else:
            return {
                'status': status_result,
                'occupation': occupation_result
            }
    
    def predict_side_specific(self, left_image_path, right_image_path, return_probabilities=False):
        """Get predictions for both left and right sides"""
        left_predictions = self.predict_combined(left_image_path, return_probabilities)
        right_predictions = self.predict_combined(right_image_path, return_probabilities)
        
        return {
            'left': left_predictions,
            'right': right_predictions
        }
    
    def get_imperfect_prediction(self, image_path, accuracy_rate=0.5):
        """
        Get intentionally imperfect predictions for user interface
        accuracy_rate: 0.4-0.6 for 40-60% accuracy
        """
        true_prediction = self.predict_combined(image_path)
        # Always return the true prediction (disable randomization for now)
        return {
            'status': true_prediction['status'],
            'occupation': true_prediction['occupation'],
            'is_correct': True
        }
    
    def get_rl_observation_vector(self, left_image_path, right_image_path):
        """
        Get observation vector for RL agent (simplified: 4 values)
        Returns predicted class indices for both sides: [left_status_idx, left_occ_idx, right_status_idx, right_occ_idx]
        """
        left_pred = self.predict_combined(left_image_path, return_probabilities=True)
        right_pred = self.predict_combined(right_image_path, return_probabilities=True)
        
        # Extract predicted class indices
        left_status_idx = np.argmax(left_pred['status_probabilities'])
        left_occupation_idx = np.argmax(left_pred['occupation_probabilities'])
        right_status_idx = np.argmax(right_pred['status_probabilities'])
        right_occupation_idx = np.argmax(right_pred['occupation_probabilities'])
        
        return np.array([left_status_idx, left_occupation_idx, right_status_idx, right_occupation_idx])
    
    def get_status_probabilities_for_rl(self, left_image_path, right_image_path):
        """
        Get status probabilities for both sides (backward compatibility with existing RL)
        Returns: [left_status_probs, right_status_probs] -> 8 values total
        """
        left_status, left_probs = self.predict_status(left_image_path, return_probabilities=True)
        right_status, right_probs = self.predict_status(right_image_path, return_probabilities=True)
        
        return np.concatenate([left_probs, right_probs])


class ImperfectCNNDisplay:
    """
    Handles the user-facing imperfect CNN display
    Shows predictions above images with configurable accuracy
    """
    
    def __init__(self, predictor, accuracy_rate=0.5):
        self.predictor = predictor
        self.accuracy_rate = accuracy_rate
    
    def set_accuracy(self, accuracy_rate):
        """Set the accuracy rate for imperfect predictions"""
        self.accuracy_rate = max(0.3, min(0.7, accuracy_rate))  # Clamp between 30-70%
    
    def get_display_text(self, image_path, side=""):
        """Get text to display above image"""
        try:
            prediction = self.predictor.get_imperfect_prediction(image_path, self.accuracy_rate)
            
            if prediction is None:
                side_text = "I think left is:" if side.upper() == "LEFT" else "I think right is:"
                return f"{side_text} Unknown"
            
            status = prediction['status'].title()
            occupation = prediction['occupation']
            
            # Format display text with conversational phrasing
            side_text = "I think left is:" if side.upper() == "LEFT" else "I think right is:"
            display_text = f"{side_text} {status} {occupation}"
            
            return display_text
        except Exception as e:
            print(f"Error getting display text for {image_path}: {e}")
            side_text = "I think left is:" if side.upper() == "LEFT" else "I think right is:"
            return f"{side_text} Error"
    
    def get_both_sides_display(self, left_image_path, right_image_path):
        """Get display text for both sides"""
        left_text = self.get_display_text(left_image_path, "Left")
        right_text = self.get_display_text(right_image_path, "Right")
        
        return left_text, right_text


# Singleton instance for global access
_enhanced_predictor = None

def get_enhanced_predictor():
    """Get global enhanced predictor instance"""
    global _enhanced_predictor
    if _enhanced_predictor is None:
        _enhanced_predictor = EnhancedPredictor()
    return _enhanced_predictor
