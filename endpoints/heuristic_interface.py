import os
import math
import random
import tkinter as tk
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

from gameplay.enums import ActionCost, State
from gameplay.humanoid import Humanoid
from models.DefaultCNN import DefaultCNN
from models.TransferStatusCNN import TransferStatusCNN
from endpoints.enhanced_predictor import EnhancedPredictor

import warnings


class Predictor(object):
    def __init__(self, classes=3, model_file=os.path.join('models', 'transfer_status_baseline.pth')):
        self.classes = classes
        self.net = None
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.is_model_loaded: bool = self._load_model(model_file, classes)
        if not self.is_model_loaded:
            warnings.warn("Model not loaded, resorting to random prediction")

    def _load_model(self, weights_path, num_classes=3):
        try:
            # Load the TransferStatusCNN model (3-class system)
            self.net = TransferStatusCNN(num_classes=3)
            state_dict = torch.load(weights_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            self.net.to(self.device)
            self.net.eval()
            print("✅ Loaded TransferStatusCNN model successfully (3-class system)")
            return True
        except Exception as e:
            print(f"❌ Failed to load CNN model: {e}")
            print(f"   Model path: {weights_path}")
            print(f"   Device: {self.device}")
            return False

    def get_probs(self, img_):
        """Get classification probabilities from CNN"""
        if self.is_model_loaded:
            try:
                # Ensure RGB format
                if img_.mode != 'RGB':
                    img_ = img_.convert('RGB')
                
                # Resize to expected input size (512x512) if necessary
                if img_.size != (512, 512):
                    img_ = img_.resize((512, 512), Image.Resampling.LANCZOS)
                
                img_tensor = self.transforms(img_).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.net(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, 1)[0].cpu().numpy()
                        
            except Exception as e:
                print(f"CNN prediction error: {e}")
                probs = np.ones(self.classes) / self.classes
        else:
            probs = np.ones(self.classes) / self.classes
        return probs
    
    def get_enhanced_prediction(self, img_):
        """
        Get enhanced prediction with human-readable status and occupation
        Returns: dict with 'status', 'occupation', 'confidence', 'enhanced_class'
        """
        probs = self.get_probs(img_)
        predicted_idx = np.argmax(probs)
        confidence = probs[predicted_idx]
        
        # Map 3-class indices to status
        status_classes = ['healthy', 'injured', 'zombie']
        status = status_classes[predicted_idx] if predicted_idx < len(status_classes) else 'healthy'
        
        # For now, assume civilian occupation (can be enhanced later with occupation CNN)
        occupation = 'civilian'
        
        return {
            'status': status,
            'occupation': occupation, 
            'confidence': confidence,
            'enhanced_class': f"{status}_{occupation}",
            'all_probs': probs
        }


class HeuristicInterface(object):
  
    def __init__(self, root, w, h, display=False, model_file=os.path.join('models', 'baseline.pth'),
                 img_data_root='data'):
        """
        Heuristic interface that properly simulates the real game
        Uses both status and occupation CNNs via EnhancedPredictor
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text = ""
        self.display = display
        self.img_data_root = 'data/modified_dataset'

        # Initialize both predictors for different use cases
        self.predictor = Predictor(classes=3, model_file=model_file)
        
        # Load the enhanced predictor (status + occupation CNNs)
        try:
            self.enhanced_predictor = EnhancedPredictor()
            print("✅ Enhanced predictor loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load enhanced predictor: {e}")
            self.enhanced_predictor = None
            warnings.warn("Enhanced predictor not loaded, resorting to random decisions")

        if self.display and root:
            self.canvas = tk.Canvas(root, width=math.floor(0.2 * w), height=math.floor(0.1 * h))
            self.canvas.place(x=math.floor(0.75 * w), y=math.floor(0.75 * h))
            from ui_elements.theme import UPGRADE_FONT
            self.label = tk.Label(self.canvas, text="Simon says...", font=UPGRADE_FONT)
            self.label.pack(side=tk.TOP)

            self.suggestion = tk.Label(self.canvas, text=self.text, font=UPGRADE_FONT)
            self.suggestion.pack(side=tk.TOP)

    def suggest(self, humanoid, capacity_full=False):
        """Legacy method for single humanoid suggestions"""
        if self.predictor.is_model_loaded:
            action = self._get_single_suggestion(humanoid, capacity_full)
        else:
            action = self.get_random_suggestion()
        self.text = action.name
        if self.display:
            self.suggestion.config(text=self.text)

    def act(self, scorekeeper, humanoid):
        """Legacy method for acting on single humanoid"""
        self.suggest(humanoid, scorekeeper.at_capacity())
        action = self.text
        if action == ActionCost.SKIP.name:
            scorekeeper.skip(humanoid)
        elif action == ActionCost.SQUISH.name:
            scorekeeper.squish(humanoid)
        elif action == ActionCost.SAVE.name:
            scorekeeper.save(humanoid)
        elif action == ActionCost.SCRAM.name:
            scorekeeper.scram(humanoid)
        else:
            raise ValueError("Invalid action suggested")

    def _get_single_suggestion(self, image_or_humanoid, is_capacity_full) -> ActionCost:
        """Handle single humanoid/image suggestions"""
        # Handle both Image objects (from main.py) and Humanoid objects
        if hasattr(image_or_humanoid, 'fp'):
            # It's a Humanoid object
            image_path = image_or_humanoid.fp
        else:
            # It's an Image object
            image_path = image_or_humanoid.Filename
        
        try:
            img_ = Image.open(os.path.join(self.img_data_root, os.path.basename(image_path)))
            prediction = self.predictor.get_enhanced_prediction(img_)
            predicted_status = prediction['status']
            
            # Simple mapping for single humanoid
            if is_capacity_full:
                return ActionCost.SCRAM
            elif predicted_status == 'zombie':
                return ActionCost.SQUISH
            elif predicted_status in ['healthy', 'injured']:
                return ActionCost.SAVE
            else:
                return ActionCost.SKIP
                
        except Exception as e:
            print(f"❌ Error in single suggestion: {e}")
            return self.get_random_suggestion()

    def get_model_suggestion(self, image_left, image_right, is_capacity_full) -> ActionCost:
        """
        Get AI suggestion for a junction scenario (left vs right choice)
        This simulates the real game's decision-making process
        """
        if not self.enhanced_predictor:
            return self.get_random_suggestion()
        
        try:
            # Get predictions for both sides using enhanced predictor
            left_filename = os.path.basename(image_left.Filename)
            right_filename = os.path.basename(image_right.Filename)
            left_path = os.path.join('data/modified_dataset', left_filename)
            right_path = os.path.join('data/modified_dataset', right_filename)
            
            left_prediction = self.enhanced_predictor.predict_combined(left_path)
            right_prediction = self.enhanced_predictor.predict_combined(right_path)
            
            # Extract status and occupation for both sides
            left_status = left_prediction['status']
            left_occupation = left_prediction['occupation']
            
            right_status = right_prediction['status']
            right_occupation = right_prediction['occupation']

            # 🔍 DEBUG: Enhanced CNN Accuracy Testing
            print(f"🧠 HEURISTIC DECISION - Junction Scenario:")
            print(f"   LEFT: {left_status} {left_occupation}")
            print(f"   RIGHT: {right_status} {right_occupation}")
            print(f"   CAPACITY FULL: {is_capacity_full}")
            
            # Apply heuristic decision logic
            recommended_action = self._make_heuristic_decision(
                left_status, left_occupation,
                right_status, right_occupation,
                is_capacity_full
            )
            
            print(f"   ⚡ Recommended Action: {recommended_action.name}")
            print("-" * 60)
            return recommended_action

        except Exception as e:
            print(f"❌ Error in heuristic decision: {e}")
            return self.get_random_suggestion()

    def _make_heuristic_decision(self, left_status, left_occupation,
                                right_status, right_occupation,
                                is_capacity_full) -> ActionCost:
        """
        Enhanced heuristic decision logic that considers both sides
        and prioritizes based on status and occupation
        """
        if is_capacity_full:
            return ActionCost.SCRAM
            
        # Define priority scores for different statuses and occupations
        status_priority = {
            'zombie': 0,      # Lowest priority - squish
            'injured': 3,     # High priority - save
            'healthy': 2,     # Medium priority - save
        }
        
        occupation_priority = {
            'Doctor': 3,      # Highest priority - medical expertise
            'Child': 3,       # Highest priority - vulnerable
            'Police': 2,      # High priority - law enforcement
            'Civilian': 1,    # Standard priority
        }
        
        # Calculate priority scores for each side
        def calculate_priority(status, occupation):
            status_score = status_priority.get(status, 0)
            occupation_score = occupation_priority.get(occupation, 0)
            total_score = status_score + occupation_score
            return total_score
        
        left_priority = calculate_priority(left_status, left_occupation)
        right_priority = calculate_priority(right_status, right_occupation)
        
        # Decision logic
        if left_status == 'zombie' and right_status == 'zombie':
            # Both are zombies - skip both
            return ActionCost.SKIP
        elif left_status == 'zombie':
            # Only left is zombie - save right
            return ActionCost.SAVE
        elif right_status == 'zombie':
            # Only right is zombie - save left
            return ActionCost.SAVE
        elif left_priority > right_priority:
            # Left has higher priority
            return ActionCost.SAVE
        elif right_priority > left_priority:
            # Right has higher priority
            return ActionCost.SAVE
        else:
            # Equal priority - save left by default
            return ActionCost.SAVE

    @staticmethod
    def get_random_suggestion():
        """Fallback to random decision if models fail"""
        return random.choice([
            ActionCost.SKIP,
            ActionCost.SAVE,
            ActionCost.SCRAM
        ])
