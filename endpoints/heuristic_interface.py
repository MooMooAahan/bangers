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
    def __init__(self, classes=4, model_file=os.path.join('models', 'baseline.pth')):
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

    def _load_model(self, weights_path, num_classes=4):
        try:
            # First, try loading as new multi-task model
            self.net = DefaultCNN(num_classes, legacy_mode=False)
            state_dict = torch.load(weights_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            print("✅ Loaded new multi-task CNN model")
            return True
        except Exception as e1:
            try:
                # If that fails, try loading as legacy model
                print("⚠️  New model failed, trying legacy model...")
                self.net = DefaultCNN(num_classes, legacy_mode=True)
                state_dict = torch.load(weights_path, map_location=self.device)
                self.net.load_state_dict(state_dict)
                print("✅ Loaded legacy CNN model")
                return True
            except Exception as e2:
                print(f"❌ Failed to load both new and legacy models:")
                print(f"   New model error: {e1}")
                print(f"   Legacy model error: {e2}")
                return False

    def get_probs(self, img_):
        """Get probabilities from both legacy and multi-task CNN"""
        if self.is_model_loaded:
            try:
                # Ensure RGB format
                if img_.mode != 'RGB':
                    img_ = img_.convert('RGB')
                
                img_tensor = self.transforms(img_).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.net(img_tensor)
                    
                    # Handle both legacy and new CNN outputs
                    if isinstance(outputs, dict):
                        # New multi-task CNN
                        person1_probs, person2_probs, count_probs = self.net.get_probs(outputs)
                        
                        # For backward compatibility, return average of valid person predictions
                        # Weight by count predictions
                        if count_probs[0] > 0.5:  # Likely 0 people
                            # Default to neutral/healthy distribution if no people detected
                            probs = np.array([0.1, 0.6, 0.2, 0.1])  # [zombie, healthy, injured, corpse]
                        elif count_probs[1] > 0.5:  # Likely 1 person
                            probs = person1_probs
                        else:  # Likely 2 people, take average
                            probs = (person1_probs + person2_probs) / 2
                    else:
                        # Legacy single-task CNN - simple tensor output
                        probs = torch.nn.functional.softmax(outputs, 1)[0].cpu().numpy()
                        
            except Exception as e:
                print(f"CNN prediction error: {e}")
                probs = np.ones(self.classes) / self.classes
        else:
            probs = np.ones(self.classes) / self.classes
        return probs


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
        
        img_ = Image.open(os.path.join(self.img_data_root, image_path))
        probs: np.ndarray = self.predictor.get_probs(img_)

        predicted_ind: int = np.argmax(probs, 0)
        class_string = Humanoid.get_all_states()[predicted_ind]
        predicted_state = State(class_string)

        # 🔍 DEBUG: CNN Accuracy Testing
        confidence = probs[predicted_ind]
        
        print(f"🧠 CNN DEBUG - Image: {image_path}")
        print(f"   📊 Predictions: {dict(zip(Humanoid.get_all_states(), probs))}")
        print(f"   🎯 Predicted: {predicted_state.value} (confidence: {confidence:.3f})")
        print(f"   ✅ Actual: {actual_state.value}")
        print(f"   {'✅ CORRECT' if predicted_state == actual_state else '❌ WRONG'}")
        print(f"   🏥 Occupation: {occupation}")
        print(f"   ⚡ Recommended Action: {self._map_class_to_action_default(predicted_state, is_capacity_full).name}")
        print("-" * 60)

        # given the model's class prediction, recommend an action
        recommended_action = self._map_class_to_action_default(predicted_state, is_capacity_full)
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
