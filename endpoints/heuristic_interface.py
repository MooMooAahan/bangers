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

import warnings


class Predictor(object):
    def __init__(self, classes=21, model_file=os.path.join('models', 'baseline.pth')):
        self.classes = classes
        self.net = None
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.is_model_loaded: bool = self._load_model(model_file)
        if not self.is_model_loaded:
            warnings.warn("Model not loaded, resorting to random prediction")

    def _load_model(self, weights_path, num_classes=21):
        try:
            # Load state dict first to determine FC layer size
            state_dict = torch.load(weights_path, map_location=self.device)
            
            # Extract FC layer dimensions from saved model  
            fc_input_size = state_dict['fc.weight'].shape[1]
            
            # Create model and initialize FC layer with correct size
            self.net = DefaultCNN(num_classes, input_size=512)
            self.net.fc = torch.nn.Linear(fc_input_size, num_classes)
            self.net.feature_size = fc_input_size
            
            # Load the state dict
            self.net.load_state_dict(state_dict)
            self.net.to(self.device)
            self.net.eval()
            print("✅ Loaded enhanced 21-class CNN model successfully")
            print(f"   FC layer expects {fc_input_size} features")
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
        
        # Get enhanced class name
        enhanced_class = Humanoid.int_to_enhanced_class(predicted_idx)
        
        # Parse to status and occupation
        status, occupation = Humanoid.parse_enhanced_class(enhanced_class)
        
        return {
            'status': status,
            'occupation': occupation, 
            'confidence': confidence,
            'enhanced_class': enhanced_class,
            'all_probs': probs
        }
    
    def format_prediction_text(self, prediction_dict):
        """
        Format prediction for user display
        Args:
            prediction_dict: Output from get_enhanced_prediction
        Returns:
            Human-readable string like "I think this is a zombie police"
        """
        status = prediction_dict['status']
        occupation = prediction_dict['occupation']
        confidence = prediction_dict['confidence']
        
        if status == 'no_person':
            return "I think there's no person in this image"
        
        # Handle corpse vs zombie distinction
        if status == 'corpse':
            status_text = "dead"
        else:
            status_text = status
            
        return f"I think this is a {status_text} {occupation} (confidence: {confidence:.1%})"


class HeuristicInterface(object):
    def __init__(self, root, w, h, display=False, model_file=os.path.join('models', 'baseline.pth'),
                 img_data_root='data'):
        self.text = ""
        self.display = display
        self.img_data_root = img_data_root

        # load 
        self.predictor = Predictor(model_file=model_file)

        if self.display:
            self.canvas = tk.Canvas(root, width=math.floor(0.2 * w), height=math.floor(0.1 * h))
            self.canvas.place(x=math.floor(0.75 * w), y=math.floor(0.75 * h))
            self.label = tk.Label(self.canvas, text="Simon says...", font=("Arial", 20))
            self.label.pack(side=tk.TOP)

            self.suggestion = tk.Label(self.canvas, text=self.text, font=("Arial", 20))
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
            actual_occupation = getattr(image_or_humanoid, 'role', 'Unknown')
        else:
            # It's an Image object
            image_path = image_or_humanoid.Filename
            # For Image objects, get the actual state from the first humanoid
            first_humanoid = next((h for h in image_or_humanoid.humanoids if h is not None), None)
            if first_humanoid:
                actual_state = State(first_humanoid.state)
                actual_occupation = getattr(first_humanoid, 'role', 'Unknown')
            else:
                actual_state = State.HEALTHY  # Default fallback
                actual_occupation = 'Unknown'
        
        img_ = Image.open(os.path.join(self.img_data_root, image_path))
        
        # Get enhanced prediction with status and occupation
        prediction = self.predictor.get_enhanced_prediction(img_)
        predicted_status = prediction['status']
        predicted_occupation = prediction['occupation']
        confidence = prediction['confidence']
        
        # Convert predicted status to State enum for action mapping
        status_map = {
            'zombie': State.ZOMBIE,
            'healthy': State.HEALTHY,
            'injured': State.INJURED,
            'corpse': State.CORPSE,
            'no_person': State.HEALTHY  # Default fallback
        }
        predicted_state = status_map.get(predicted_status, State.HEALTHY)

        # 🔍 DEBUG: Enhanced CNN Accuracy Testing
        print(f"🧠 ENHANCED CNN DEBUG - Image: {image_path}")
        print(f"   🎯 Predicted: {predicted_status} {predicted_occupation} (confidence: {confidence:.1%})")
        print(f"   ✅ Actual: {actual_state.value} {actual_occupation}")
        
        # Check if status prediction is correct (occupation comparison is informational)
        status_correct = predicted_status == actual_state.value
        print(f"   {'✅ STATUS CORRECT' if status_correct else '❌ STATUS WRONG'}")
        print(f"   🏥 Occupation Match: {predicted_occupation.lower() == actual_occupation.lower()}")
        print(f"   💬 Display Text: \"{self.predictor.format_prediction_text(prediction)}\"")
        print(f"   ⚡ Recommended Action: {self._map_class_to_action_enhanced(predicted_status, predicted_occupation, is_capacity_full).name}")
        print("-" * 60)

        # Given the enhanced prediction, recommend an action
        recommended_action = self._map_class_to_action_enhanced(predicted_status, predicted_occupation, is_capacity_full)
        return recommended_action

    @staticmethod
    def _map_class_to_action_default(predicted_state: State, is_capacity_full: bool = False) -> ActionCost:
        # map prediction to ActionCost to return the right thing; now aligned with Rob's pseudocode
        if is_capacity_full:
            return ActionCost.SCRAM
        if predicted_state is State.ZOMBIE:
            return ActionCost.SQUISH
        if predicted_state is State.INJURED:
            return ActionCost.SAVE
        if predicted_state is State.HEALTHY:
            return ActionCost.SAVE
        if predicted_state is State.CORPSE:
            return ActionCost.SQUISH
    
    @staticmethod
    def _map_class_to_action_enhanced(predicted_status: str, predicted_occupation: str, is_capacity_full: bool = False) -> ActionCost:
        """
        Enhanced action mapping that considers both status and occupation
        Priority decisions for special occupations (doctors, children)
        """
        if is_capacity_full:
            return ActionCost.SCRAM
            
        # Zombies and corpses should be squished regardless of occupation
        if predicted_status in ['zombie', 'corpse']:
            return ActionCost.SQUISH
            
        # For living people (healthy/injured), consider occupation priority
        if predicted_status in ['healthy', 'injured']:
            # High priority saves: doctors (medical expertise) and children (vulnerable)
            if predicted_occupation in ['doctor', 'child']:
                return ActionCost.SAVE
            # Medium priority: police (law enforcement value), civilians
            elif predicted_occupation in ['police', 'civilian']:
                return ActionCost.SAVE
            # Lower priority: militants (potential threat, but still human)
            elif predicted_occupation == 'militant':
                return ActionCost.SAVE  # Still save them, but noted
            else:
                return ActionCost.SAVE  # Default: save living people
        
        # Default fallback
        return ActionCost.SKIP
