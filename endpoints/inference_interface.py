import os
import math
import random
import tkinter as tk
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

from gameplay.enums import ActionCost, State
from gameplay.scorekeeper import ScoreKeeper
from gameplay.humanoid import Humanoid

from models.PPO import ActorCritic, PPO
# Predictor class removed from heuristic_interface

from gym import Env, spaces
from endpoints.data_parser import DataParser

import warnings

class RLPredictor(object):
    def __init__(self,
                 actions = 6,  # Updated: SKIP_BOTH, SQUISH_LEFT, SQUISH_RIGHT, SAVE_LEFT, SAVE_RIGHT, SCRAM
                 model_file=os.path.join('models', 'baselineRL.pth'),
                 img_data_root='./data'):
        self.actions = actions
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
            self.net = PPO(0,0,0,0,0,False,0.6)
            self.net.load(weights_path)
            return True
        except Exception as e:  # file not found, maybe others?
            print(e)
            return False
    def get_action(self, observation_space):
        if self.is_model_loaded:
            action = self.net.select_action(observation_space)
        else:
            action = np.random.randint(0, self.actions)
        return action
    
class InferInterface(Env):
    def __init__(self, root, w, h, data_parser, scorekeeper, 
                 classifier_model_file=os.path.join('models', 'transfer_status_baseline.pth'), 
                 rl_model_file=os.path.join('models', 'baselineRL.pth'), 
                 img_data_root='data', display=False):
        """
        initializes RL inference interface
        
        dataparser : stores humanoid information needed to retreive humanoid images and rewards
        scorekeeper : keeps track of actions being done on humanoids, score, and is needed for reward calculations
        classifier_model_file : backbone model weights used in RL observation state
        rl_model_file : trained RL model for action prediction
        """
        self.img_data_root = img_data_root
        self.data_parser = data_parser
        self.scorekeeper = scorekeeper
        self.display = display

        self.environment_params = {
            "car_capacity" : self.scorekeeper.capacity,
            "num_classes" : len(Humanoid.get_all_states()),
            "num_actions" : 6,  # Updated: 6 actions like training interface
        }
        
        # Initialize observation space matching training interface
        self.observation_space = {
            "variables": np.zeros(4),  # time, reward, capacity, zombie_count
            "vehicle_storage_class_probs": np.zeros((self.environment_params['car_capacity'], self.environment_params['num_classes'])),
            "humanoid_class_probs": np.zeros(self.environment_params['num_classes'] * 2),  # Both left and right
            "doable_actions": np.ones(self.environment_params['num_actions'], dtype=np.int64),
        }

        self.action_space = spaces.Discrete(self.environment_params['num_actions'])
        
                # Initialize enhanced predictor (replaces old Predictor)
        self.enhanced_predictor = EnhancedPredictor(
            status_model_file=classifier_model_file,
            occupation_model_file='models/optimized_4class_occupation.pth'
        )
        self.action_predictor = RLPredictor(actions=self.environment_params['num_actions'],
                                          model_file=rl_model_file)
        
        # Initialize state variables
        self.current_image_left = None
        self.current_image_right = None
        self.current_humanoid_probs_left = np.ones(self.environment_params['num_classes']) / self.environment_params['num_classes']
        self.current_humanoid_probs_right = np.ones(self.environment_params['num_classes']) / self.environment_params['num_classes']
        self.previous_cum_reward = 0
        
        self.reset()

        if self.display:
            self.canvas = tk.Canvas(root, width=math.floor(0.2 * w), height=math.floor(0.1 * h))
            self.canvas.place(x=math.floor(0.75 * w), y=math.floor(0.75 * h))
            from ui_elements.theme import UPGRADE_FONT
            self.label = tk.Label(self.canvas, text="RL Agent Inference...", font=UPGRADE_FONT)
            self.label.pack(side=tk.TOP)

            self.suggestion = tk.Label(self.canvas, text="", font=UPGRADE_FONT)
            self.suggestion.pack(side=tk.TOP)
            
    def reset(self):
        """
        resets game for a new episode to run.
        returns observation space
        """
        self.observation_space = {
            "variables": np.zeros(4),  # time, reward, capacity, zombie_count
            "vehicle_storage_class_probs": np.zeros((self.environment_params['car_capacity'], self.environment_params['num_classes'])),
            "humanoid_class_probs": np.zeros(self.environment_params['num_classes'] * 2),  # Both left and right
            "doable_actions": np.ones(self.environment_params['num_actions'], dtype=np.int64),
        }
        self.previous_cum_reward = 0
        self.data_parser.reset()
        self.scorekeeper.reset()
        
        # Get initial images and set up environment
        self.get_current_images()
        self.get_observation_space()
        
        return self.observation_space
    
    def get_current_images(self):
        """
        Gets both left and right images from the dataparser to match actual game mechanics
        """
        try:
            # Get both left and right images like the actual game
            self.current_image_left = self.data_parser.get_random(side='left')
            self.current_image_right = self.data_parser.get_random(side='right')
            
            # Load both images for CNN prediction
            img_path_left = os.path.join(self.img_data_root, self.current_image_left.Filename)
            img_path_right = os.path.join(self.img_data_root, self.current_image_right.Filename)
            
            pil_img_left = Image.open(img_path_left)
            pil_img_right = Image.open(img_path_right)
            
            # Use enhanced predictor to get status and occupation predictions
            left_prediction = self.enhanced_predictor.predict_combined(self.current_image_left.Filename)
            right_prediction = self.enhanced_predictor.predict_combined(self.current_image_right.Filename)
            
            # Extract status and occupation indices for RL observation
            self.current_humanoid_probs_left = np.array([
                ['healthy', 'injured', 'zombie'].index(left_prediction['status']),
                ['Civilian', 'Child', 'Doctor', 'Police'].index(left_prediction['occupation'])
            ])
            self.current_humanoid_probs_right = np.array([
                ['healthy', 'injured', 'zombie'].index(right_prediction['status']),
                ['Civilian', 'Child', 'Doctor', 'Police'].index(right_prediction['occupation'])
            ])
            
        except Exception as e:
            print(f"Error loading images: {e}")
            # Fallback to random status and occupation indices
            self.current_humanoid_probs_left = np.array([0, 0])  # [status_idx, occupation_idx]
            self.current_humanoid_probs_right = np.array([0, 0])  # [status_idx, occupation_idx]
    
    def get_observation_space(self):
        """
        Updates the observation space with current game state including both left and right scenarios
        """
        # Normalize values for better RL training
        zombie_count_normalized = min(self.scorekeeper.ambulance.get("zombie", 0) / self.scorekeeper.capacity, 1.0)
        self.observation_space['variables'] = np.array([
            self.scorekeeper.remaining_time / self.scorekeeper.shift_len,  # Normalize to [0,1]
            self.previous_cum_reward / 100.0,  # Scale reward
            sum(self.scorekeeper.ambulance.values()) / self.scorekeeper.capacity,  # Capacity ratio
            zombie_count_normalized,  # Zombie ratio (important for police effect)
        ])
        
        self.observation_space["doable_actions"] = np.array(self.scorekeeper.available_action_space(), dtype=np.int64)
        
        # Include both left and right humanoid probabilities
        combined_probs = np.concatenate([self.current_humanoid_probs_left, self.current_humanoid_probs_right])
        self.observation_space["humanoid_class_probs"] = combined_probs
        
        # Update vehicle storage probabilities based on current ambulance contents
        current_capacity = self.scorekeeper.get_current_capacity()
        for i in range(current_capacity):
            if i < len(self.observation_space["vehicle_storage_class_probs"]):
                total_in_ambulance = sum(self.scorekeeper.ambulance.values())
                if total_in_ambulance > 0:
                    self.observation_space["vehicle_storage_class_probs"][i] = np.array([
                        self.scorekeeper.ambulance.get("zombie", 0) / total_in_ambulance,
                        self.scorekeeper.ambulance.get("healthy", 0) / total_in_ambulance, 
                        self.scorekeeper.ambulance.get("injured", 0) / total_in_ambulance,
                        0  # corpse placeholder
                    ])
        
        return self.observation_space
    
    def act(self, humanoid=None):
        """
        Acts on the environment using RL agent decision-making
        Gets current left/right images and makes action based on observation state
        """
        try:
            # Get current images and probabilities
            self.get_current_images()
            
            # Get RL action based on current observation
            action_idx = self.action_predictor.get_action(self.get_observation_space())
            
            # Execute action based on new 6-action space
            # Actions: 0=SKIP_BOTH, 1=SQUISH_LEFT, 2=SQUISH_RIGHT, 3=SAVE_LEFT, 4=SAVE_RIGHT, 5=SCRAM
            if action_idx == 0:  # SKIP_BOTH
                if not (self.scorekeeper.remaining_time <= 0):
                    self.scorekeeper.skip_both(self.current_image_left, self.current_image_right)
                    
            elif action_idx == 1:  # SQUISH_LEFT
                if not (self.scorekeeper.remaining_time <= 0):
                    self.scorekeeper.squish(self.current_image_left)
                    
            elif action_idx == 2:  # SQUISH_RIGHT
                if not (self.scorekeeper.remaining_time <= 0):
                    self.scorekeeper.squish(self.current_image_right)
                    
            elif action_idx == 3:  # SAVE_LEFT
                if not (self.scorekeeper.remaining_time <= 0 or self.scorekeeper.at_capacity()):
                    self.scorekeeper.save(self.current_image_left)
                    # Update vehicle storage when saving
                    current_capacity = self.scorekeeper.get_current_capacity()
                    if current_capacity > 0 and current_capacity <= len(self.observation_space["vehicle_storage_class_probs"]):
                        # Only store status as one-hot vector
                        status_idx = int(self.current_humanoid_probs_left[0])
                        one_hot = np.zeros(self.environment_params['num_classes'])
                        one_hot[status_idx] = 1.0
                        self.observation_space["vehicle_storage_class_probs"][current_capacity-1] = one_hot
                        
            elif action_idx == 4:  # SAVE_RIGHT
                if not (self.scorekeeper.remaining_time <= 0 or self.scorekeeper.at_capacity()):
                    self.scorekeeper.save(self.current_image_right)
                    # Update vehicle storage when saving
                    current_capacity = self.scorekeeper.get_current_capacity()
                    if current_capacity > 0 and current_capacity <= len(self.observation_space["vehicle_storage_class_probs"]):
                        # Only store status as one-hot vector
                        status_idx = int(self.current_humanoid_probs_right[0])
                        one_hot = np.zeros(self.environment_params['num_classes'])
                        one_hot[status_idx] = 1.0
                        self.observation_space["vehicle_storage_class_probs"][current_capacity-1] = one_hot
                        
            elif action_idx == 5:  # SCRAM
                self.scorekeeper.scram(self.current_image_left, self.current_image_right)
                # Clear vehicle storage when scramming
                self.observation_space["vehicle_storage_class_probs"] = np.zeros((self.environment_params['car_capacity'], self.environment_params['num_classes']))
                
            else:
                print(f"Invalid action index: {action_idx}")
            
            # Update reward tracking
            current_reward = self.scorekeeper.get_cumulative_reward()
            self.previous_cum_reward = current_reward
            
            # Update display if enabled
            if self.display:
                action_names = ["SKIP_BOTH", "SQUISH_LEFT", "SQUISH_RIGHT", "SAVE_LEFT", "SAVE_RIGHT", "SCRAM"]
                self.suggestion.config(text=f"Action: {action_names[action_idx]}")
                
        except Exception as e:
            print(f"Error in inference act: {e}")
            # Fallback action
            if not self.scorekeeper.at_capacity():
                self.scorekeeper.skip_both(self.current_image_left, self.current_image_right)
    
    def suggest(self):
        """
        Suggests an action for the current left/right scenario using RL agent
        Returns action name as string for display purposes
        """
        try:
            # Get current images and probabilities
            self.get_current_images()
            
            # Get RL action based on current observation
            action_idx = self.action_predictor.get_action(self.get_observation_space())
            
            # Map action index to string
            action_names = ["SKIP_BOTH", "SQUISH_LEFT", "SQUISH_RIGHT", "SAVE_LEFT", "SAVE_RIGHT", "SCRAM"]
            return action_names[action_idx] if 0 <= action_idx < len(action_names) else "SKIP_BOTH"
            
        except Exception as e:
            print(f"Error in inference suggest: {e}")
            return "SKIP_BOTH"  # Default fallback action