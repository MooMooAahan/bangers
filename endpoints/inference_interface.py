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
from endpoints.enhanced_predictor import EnhancedPredictor

from gym import Env, spaces
from endpoints.data_parser import DataParser

import warnings

class RLPredictor(object):
    def __init__(self,
                 actions = 6,  # Current system has 6 actions
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
        except Exception as e:  
            print(f"Model loading error: {e}")
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
                 rl_model_file=None,  # Now requires explicit model path
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
        
                # Initialize observation space EXACTLY matching training interface (18 dimensions total)
        self.observation_space = {
            "variables": np.zeros(4),  # [time_ratio, reward_scaled, capacity_ratio, zombie_ratio]
            "vehicle_storage_summary": np.zeros(4),  # [zombie_ratio, healthy_ratio, injured_ratio, corpse_ratio]
            "humanoid_class_probs": np.zeros(4),  # 4 values: [left_status, left_occ, right_status, right_occ]
            "doable_actions": np.ones(self.environment_params['num_actions'], dtype=np.int64),
        }

        self.action_space = spaces.Discrete(self.environment_params['num_actions'])
        
                # Initialize enhanced predictor (replaces old Predictor)
        self.enhanced_predictor = EnhancedPredictor(
            status_model_file=classifier_model_file,
            occupation_model_file='models/optimized_4class_occupation.pth'
        )
        # Require explicit model path - no more default fallback to wrong model
        if rl_model_file is None:
            raise ValueError("rl_model_file must be specified - no default model to avoid architecture mismatches")
        
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
            "variables": np.zeros(4),  # [time_ratio, reward_scaled, capacity_ratio, zombie_ratio]
            "vehicle_storage_summary": np.zeros(4),  # [zombie_ratio, healthy_ratio, injured_ratio, corpse_ratio]
            "humanoid_class_probs": np.zeros(4),  # 4 values: [left_status, left_occ, right_status, right_occ]
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
            
            # Extract status and occupation indices for RL observation (EXACT training format)
            try:
                left_status_idx = ['healthy', 'injured', 'zombie'].index(left_prediction['status'])
            except ValueError:
                left_status_idx = 0  # Default to healthy
            try:
                left_occupation_idx = ['Civilian', 'Child', 'Doctor', 'Police'].index(left_prediction['occupation'])
            except ValueError:
                left_occupation_idx = 0  # Default to Civilian
                
            try:
                right_status_idx = ['healthy', 'injured', 'zombie'].index(right_prediction['status'])
            except ValueError:
                right_status_idx = 0
            try:
                right_occupation_idx = ['Civilian', 'Child', 'Doctor', 'Police'].index(right_prediction['occupation'])
            except ValueError:
                right_occupation_idx = 0
                
            self.current_humanoid_probs_left = np.array([left_status_idx, left_occupation_idx])
            self.current_humanoid_probs_right = np.array([right_status_idx, right_occupation_idx])
            
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
            self.scorekeeper.remaining_time / self.scorekeeper.shift_len,  # Time ratio [0,1]
            np.clip(self.previous_cum_reward / 100.0, -5.0, 5.0),  # Scaled reward (clipped for stability)
            sum(self.scorekeeper.ambulance.values()) / self.scorekeeper.capacity,  # Capacity ratio [0,1]
            zombie_count_normalized,  # Zombie ratio [0,1] (important for police effect)
        ])
        
        # Use fixed 6-action space to match training (not dynamic scorekeeper actions)
        self.observation_space["doable_actions"] = np.ones(self.environment_params['num_actions'], dtype=np.int64)
        
        # Include both left and right humanoid info (EXACT training format)
        # [left_status, left_occ, right_status, right_occ] = 4 values total
        combined_info = np.concatenate([self.current_humanoid_probs_left, self.current_humanoid_probs_right])
        self.observation_space["humanoid_class_probs"] = combined_info
        
        # Update vehicle storage summary (matching training interface format)
        total_in_ambulance = sum(self.scorekeeper.ambulance.values())
        if total_in_ambulance > 0:
            # Aggregate ambulance composition into 4 values total
            vehicle_summary = np.array([
                self.scorekeeper.ambulance.get("zombie", 0) / max(1, total_in_ambulance),
                self.scorekeeper.ambulance.get("healthy", 0) / max(1, total_in_ambulance), 
                self.scorekeeper.ambulance.get("injured", 0) / max(1, total_in_ambulance),
                0  # corpse placeholder
            ])
        else:
            vehicle_summary = np.zeros(4)
        
        # Store as single summary vector 
        self.observation_space["vehicle_storage_summary"] = vehicle_summary
        
        return self.observation_space
    
    def _update_predictions_from_metadata(self):
        """Update humanoid predictions using ground truth metadata instead of CNN"""
        def get_ground_truth_indices(image):
            """Extract ground truth status and occupation indices from image metadata"""
            if hasattr(image, 'humanoids') and image.humanoids and image.humanoids[0] is not None:
                humanoid = image.humanoids[0]  # First humanoid in the image
                
                # Status mapping: zombie=0, healthy=1, injured=2, corpse=3
                status_map = {"zombie": 0, "healthy": 1, "injured": 2, "corpse": 3}
                status_idx = status_map.get(humanoid.state, 1)  # Default to healthy if unknown
                
                # Occupation mapping: civilian=0, child=1, doctor=2, police=3, militant=4  
                occupation_map = {"civilian": 0, "child": 1, "doctor": 2, "police": 3, "militant": 4}
                occupation_idx = occupation_map.get(humanoid.role, 0)  # Default to civilian if unknown
                
                return status_idx, occupation_idx
            else:
                return 1, 0  # Default: healthy civilian if no humanoid
        
        # Get ground truth for both sides
        left_status_idx, left_occupation_idx = get_ground_truth_indices(self.current_image_left)
        right_status_idx, right_occupation_idx = get_ground_truth_indices(self.current_image_right)
        
        self.current_humanoid_probs_left = np.array([left_status_idx, left_occupation_idx])
        self.current_humanoid_probs_right = np.array([right_status_idx, right_occupation_idx])
    
    def act(self, humanoid=None):
        """
        Acts on the environment using RL agent decision-making
        Gets current left/right images and makes action based on observation state
        """
        try:
            # Use provided humanoid or get random images if none provided
            if humanoid is not None:
                self.current_image_left = humanoid
                self.current_image_right = self.data_parser.get_random(side='right')
                # Use ground truth metadata instead of CNN predictions
                self._update_predictions_from_metadata()
            else:
                # Fallback: Get current images and probabilities
                self.get_current_images()
            
            # Get RL action based on current observation
            action_idx = self.action_predictor.get_action(self.get_observation_space())
            
            # Execute action based on new 6-action space
            # Actions: 0=SKIP_BOTH, 1=SQUISH_LEFT, 2=SQUISH_RIGHT, 3=SAVE_LEFT, 4=SAVE_RIGHT, 5=SCRAM
            if action_idx == 0:  # SKIP_BOTH
                if not (self.scorekeeper.remaining_time <= 0):
                    self.scorekeeper.skip_both(self.current_image_left, self.current_image_right)
                    # Log RL decision with CNN predictions and true classes
                    self.scorekeeper.log_consolidated('skip_both', 
                                                    image_left=self.current_image_left,
                                                    image_right=self.current_image_right,
                                                    action_side='both')
                    
            elif action_idx == 1:  # SQUISH_LEFT
                if not (self.scorekeeper.remaining_time <= 0):
                    self.scorekeeper.squish(self.current_image_left)
                    self.scorekeeper.log_consolidated('squish', 
                                                    image_left=self.current_image_left,
                                                    image_right=self.current_image_right,
                                                    action_side='left')
                    
            elif action_idx == 2:  # SQUISH_RIGHT
                if not (self.scorekeeper.remaining_time <= 0):
                    self.scorekeeper.squish(self.current_image_right)
                    self.scorekeeper.log_consolidated('squish', 
                                                    image_left=self.current_image_left,
                                                    image_right=self.current_image_right,
                                                    action_side='right')
                    
            elif action_idx == 3:  # SAVE_LEFT
                if not (self.scorekeeper.remaining_time <= 0 or self.scorekeeper.at_capacity()):
                    self.scorekeeper.save(self.current_image_left)
                    self.scorekeeper.log_consolidated('save', 
                                                    image_left=self.current_image_left,
                                                    image_right=self.current_image_right,
                                                    action_side='left')
                        
            elif action_idx == 4:  # SAVE_RIGHT
                if not (self.scorekeeper.remaining_time <= 0 or self.scorekeeper.at_capacity()):
                    self.scorekeeper.save(self.current_image_right)
                    self.scorekeeper.log_consolidated('save', 
                                                    image_left=self.current_image_left,
                                                    image_right=self.current_image_right,
                                                    action_side='right')
                        
            elif action_idx == 5:  # SCRAM
                self.scorekeeper.scram(self.current_image_left, self.current_image_right)
                # Clear vehicle storage when scramming
                self.observation_space["vehicle_storage_summary"] = np.zeros(4)
                self.scorekeeper.log_consolidated('scram', 
                                                image_left=self.current_image_left,
                                                image_right=self.current_image_right,
                                                action_side='both')
                
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