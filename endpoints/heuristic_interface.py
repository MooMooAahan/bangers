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
        self.img_data_root = img_data_root

        # load 
        self.predictor = Predictor(model_file=model_file)

        if self.display:
            self.canvas = tk.Canvas(root, width=math.floor(0.2 * w), height=math.floor(0.1 * h))
            self.canvas.place(x=math.floor(0.75 * w), y=math.floor(0.75 * h))
            from ui_elements.theme import UPGRADE_FONT
            self.label = tk.Label(self.canvas, text="Simon says...", font=UPGRADE_FONT)
            self.label.pack(side=tk.TOP)

            self.suggestion = tk.Label(self.canvas, text=self.text, font=UPGRADE_FONT)
            self.suggestion.pack(side=tk.TOP)

    def _load_model(self, weights_path, num_classes=4):
        try:
            self.net = DefaultCNN(num_classes)
            self.net.load_state_dict(torch.load(weights_path))
            return True
        except:  # file not found, maybe others?
            return False

    def suggest(self, humanoid, capacity_full=False):
        if self.predictor.is_model_loaded:
            action = self.get_model_suggestion(humanoid, capacity_full)
        else:
            action = self.get_random_suggestion()
        self.text = action.name
        if self.display:
            self.suggestion.config(text=self.text)

    def act(self, scorekeeper, humanoid):
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

    @staticmethod
    def get_random_suggestion():
        return random.choice(list(ActionCost))

    def get_model_suggestion(self, image_or_humanoid, is_capacity_full) -> ActionCost:
        # Handle both Image objects (from main.py) and Humanoid objects
        if hasattr(image_or_humanoid, 'fp'):
            # It's a Humanoid object
            image_path = image_or_humanoid.fp
            actual_state = State(image_or_humanoid.state)
            occupation = getattr(image_or_humanoid, 'role', 'Unknown')
        else:
            # It's an Image object
            image_path = image_or_humanoid.Filename
            # For Image objects, get the actual state from the first humanoid
            first_humanoid = next((h for h in image_or_humanoid.humanoids if h is not None), None)
            if first_humanoid:
                actual_state = State(first_humanoid.state)
                occupation = getattr(first_humanoid, 'role', 'Unknown')
            else:
                actual_state = State.HEALTHY  # Default fallback
                occupation = 'Unknown'
        
        # Load the enhanced predictor (status + occupation CNNs)
        try:
            self.enhanced_predictor = EnhancedPredictor()
            print("✅ Enhanced predictor loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load enhanced predictor: {e}")
            self.enhanced_predictor = None
            warnings.warn("Enhanced predictor not loaded, resorting to random decisions")

    def get_model_suggestion(self, image_left, image_right, is_capacity_full) -> ActionCost:
        """
        Get AI suggestion for a junction scenario (left vs right choice)
        This simulates the real game's decision-making process
        """
        if not self.enhanced_predictor:
            return self.get_random_suggestion()
        
        try:
            # Get predictions for both sides using enhanced predictor
            left_prediction = self.enhanced_predictor.predict_combined(image_left.Filename)
            right_prediction = self.enhanced_predictor.predict_combined(image_right.Filename)
            
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
