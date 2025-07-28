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
from models.DefaultCNN import DefaultCNN
from endpoints.heuristic_interface import Predictor
from endpoints.enhanced_predictor import EnhancedPredictor

from gym import Env, spaces
from endpoints.data_parser import DataParser


class TrainInterface(Env):
    def __init__(self, root=None, w=800, h=600, data_parser=None, scorekeeper=None, 
                 classifier_model_file=os.path.join('models', 'transfer_status_baseline.pth'), 
                 img_data_root='data', display=False):
        """
        initializes RL training interface
        
        dataparser : stores humanoid information needed to retreive humanoid images and rewards
        scorekeeper : keeps track of actions being done on humanoids, score, and is needed for reward calculations
        classifier_model_file : backbone model weights used in RL observation state
        """
        self.img_data_root = img_data_root
        self.data_parser = data_parser if data_parser else DataParser(img_data_root)
        self.scorekeeper = scorekeeper if scorekeeper else ScoreKeeper(shift_len=480, capacity=10, display=display)
        self.display = display

        self.environment_params = {
            "car_capacity" : self.scorekeeper.capacity,
            "num_classes" : len(Humanoid.get_all_states()),
            "num_actions" : 6,  # Updated: SKIP_BOTH, SQUISH_LEFT, SQUISH_RIGHT, SAVE_LEFT, SAVE_RIGHT, SCRAM
        }
        
        # Initialize observation space structure
        # Variables: [time_ratio, reward_scaled, capacity_ratio, zombie_ratio]
        # Simplified: humanoid_class_probs now includes status + occupation for each side
        # Left: [status(1), occupation(1)] + Right: [status(1), occupation(1)] = 4 total
        self.observation_space = {
            "variables": np.zeros(4),
            "vehicle_storage_class_probs": np.zeros((self.environment_params['car_capacity'], self.environment_params['num_classes'])),
            "humanoid_class_probs": np.zeros(4),  # Simplified: 4 values (2 per side: status + occupation)
            "doable_actions": np.ones(self.environment_params['num_actions'], dtype=np.int64),
        }

        self.action_space = spaces.Discrete(self.environment_params['num_actions'])
        
        # Initialize enhanced CNN predictor for observations (status + occupation)
        self.enhanced_predictor = EnhancedPredictor(
            status_model_file=classifier_model_file,
            occupation_model_file='models/optimized_4class_occupation.pth'
        )
        # Keep old predictor for backward compatibility
        self.predictor = Predictor(classes=self.environment_params['num_classes'], 
                                 model_file=classifier_model_file)
        
        # Initialize state
        self.current_image_left = None
        self.current_image_right = None
        # Simplified: Store status and occupation for each side
        self.current_humanoid_probs_left = np.array([0, 0])  # [status_idx, occupation_idx]
        self.current_humanoid_probs_right = np.array([0, 0])  # [status_idx, occupation_idx]
        # For backward compatibility
        self.current_status_probs_left = np.ones(self.environment_params['num_classes']) / self.environment_params['num_classes']
        self.current_status_probs_right = np.ones(self.environment_params['num_classes']) / self.environment_params['num_classes']
        
        self.reset()

        if self.display and root:
            self.canvas = tk.Canvas(root, width=math.floor(0.2 * w), height=math.floor(0.1 * h))
            self.canvas.place(x=math.floor(0.75 * w), y=math.floor(0.75 * h))
            self.label = tk.Label(self.canvas, text="RL Agent Training...", font=("Arial", 20))
            self.label.pack(side=tk.TOP)

            self.suggestion = tk.Label(self.canvas, text="", font=("Arial", 20))
            self.suggestion.pack(side=tk.TOP)
            
    def reset(self):
        """
        resets game for a new episode to run.
        returns observation space
        """
        # Reset observation space - simplified with status + occupation
        self.observation_space = {
            "variables": np.zeros(4),  # [time_ratio, reward_scaled, capacity_ratio, zombie_ratio]
            "vehicle_storage_class_probs": np.zeros((self.environment_params['car_capacity'], self.environment_params['num_classes'])),
            "humanoid_class_probs": np.zeros(4),  # Simplified: 4 values (2 per side: status + occupation)
            "doable_actions": np.ones(self.environment_params['num_actions'], dtype=np.int64),
        }
        
        self.previous_cum_reward = 0
        self.data_parser.reset()
        self.scorekeeper.reset()
        
        # Get initial image and set up environment
        self.get_current_image()
        self.get_observation_space()
        
        return self.observation_space
    
    def get_current_image(self):
        """
        Gets both left and right images from the dataparser to match actual game mechanics
        """
        try:
            # Get both left and right images like the actual game
            self.current_image_left = self.data_parser.get_random(side='left')
            self.current_image_right = self.data_parser.get_random(side='right')
            
            # Handle both Image objects (with .Filename) and Humanoid objects (with .fp)
            def get_image_path(image_obj):
                if hasattr(image_obj, 'fp'):
                    # It's a Humanoid object
                    return image_obj.fp
                else:
                    # It's an Image object  
                    return image_obj.Filename
            
            img_path_left = os.path.join(self.img_data_root, get_image_path(self.current_image_left))
            img_path_right = os.path.join(self.img_data_root, get_image_path(self.current_image_right))
            
            # Get simplified predictions (status + occupation for both sides)
            left_pred = self.enhanced_predictor.predict_combined(img_path_left, return_probabilities=True)
            right_pred = self.enhanced_predictor.predict_combined(img_path_right, return_probabilities=True)
            
            # Extract status and occupation indices (simplified: just the predicted class)
            left_status_idx = np.argmax(left_pred['status_probabilities'])
            left_occupation_idx = np.argmax(left_pred['occupation_probabilities'])
            right_status_idx = np.argmax(right_pred['status_probabilities'])
            right_occupation_idx = np.argmax(right_pred['occupation_probabilities'])
            
            self.current_humanoid_probs_left = np.array([left_status_idx, left_occupation_idx])
            self.current_humanoid_probs_right = np.array([right_status_idx, right_occupation_idx])
            
            # For backward compatibility, keep status-only probabilities
            self.current_status_probs_left = left_pred['status_probabilities']
            self.current_status_probs_right = right_pred['status_probabilities']
            
        except Exception as e:
            print(f"Error loading images: {e}")
            # Fallback to random indices
            self.current_humanoid_probs_left = np.array([0, 0])  # [status_idx, occupation_idx]
            self.current_humanoid_probs_right = np.array([0, 0])  # [status_idx, occupation_idx]
            self.current_status_probs_left = np.ones(self.environment_params['num_classes']) / self.environment_params['num_classes']
            self.current_status_probs_right = np.ones(self.environment_params['num_classes']) / self.environment_params['num_classes']
    
    def get_observation_space(self):
        """
        Updates the observation space with current game state including both left and right scenarios
        """        
        self.observation_space["doable_actions"] = np.array(self.scorekeeper.available_action_space(), dtype=np.int64)
        
        # Include both left and right humanoid info (simplified: status + occupation)
        # Concatenate left and right info to give agent full scenario information
        combined_info = np.concatenate([self.current_humanoid_probs_left, self.current_humanoid_probs_right])
        self.observation_space["humanoid_class_probs"] = combined_info
        
        # Update vehicle storage probabilities based on current ambulance contents
        current_capacity = self.scorekeeper.get_current_capacity()
        for i in range(current_capacity):
            if i < len(self.observation_space["vehicle_storage_class_probs"]):
                # For now, use simplified representation based on ambulance composition
                total_in_ambulance = sum(self.scorekeeper.ambulance.values())
                if total_in_ambulance > 0:
                    self.observation_space["vehicle_storage_class_probs"][i] = np.array([
                        self.scorekeeper.ambulance.get("zombie", 0) / total_in_ambulance,
                        self.scorekeeper.ambulance.get("healthy", 0) / total_in_ambulance, 
                        self.scorekeeper.ambulance.get("injured", 0) / total_in_ambulance,
                        0  # corpse placeholder
                    ])
        
        # Strategic information: consistent 4-element variables array
        zombie_count_normalized = min(self.scorekeeper.ambulance.get("zombie", 0) / self.scorekeeper.capacity, 1.0)
        self.observation_space['variables'] = np.array([
            self.scorekeeper.remaining_time / self.scorekeeper.shift_len,  # Time ratio [0,1]
            np.clip(self.previous_cum_reward / 100.0, -5.0, 5.0),  # Scaled reward (clipped for stability)
            sum(self.scorekeeper.ambulance.values()) / self.scorekeeper.capacity,  # Capacity ratio [0,1]
            zombie_count_normalized,  # Zombie ratio [0,1] (important for police effect)
        ])
        
    def step(self, action_idx):
        """
        Acts on the environment and returns the observation state, reward, etc.
        
        action_idx : the index of the action being taken
        Actions: 0=SKIP_BOTH, 1=SQUISH_LEFT, 2=SQUISH_RIGHT, 3=SAVE_LEFT, 4=SAVE_RIGHT, 5=SCRAM
        """
        
        # 🧟 Process zombie infections and cures at start of each turn (like main game)
        infected_humanoids = self.scorekeeper.process_zombie_infections()
        if infected_humanoids:
            print(f"[RL TRAINING] Zombie infection occurred: {infected_humanoids}")
            
        cured_humanoids = self.scorekeeper.process_zombie_cures()
        if cured_humanoids:
            print(f"[RL TRAINING] Zombie cure occurred: {cured_humanoids}")
        
        reward = 0
        finished = False  # is game over
        truncated = False
        
        # Execute action with proper validation and error handling
        action_executed, failure_reason = self._execute_action_with_validation(action_idx)
        
        # Provide specific feedback for failed actions
        if not action_executed:
            if failure_reason == "time_out":
                reward = -0.1  # Small penalty for time constraint
            elif failure_reason == "capacity_full":
                reward = -0.5  # Medium penalty for capacity constraint  
            elif failure_reason == "invalid_action":
                reward = -1.0  # Large penalty for invalid action
            else:
                reward = -0.3  # Generic penalty for other failures
        
        if action_executed:
            # Calculate shaped reward with immediate feedback
            reward = self._calculate_shaped_reward(action_idx)
            
            # Update cumulative tracking
            current_reward = self.scorekeeper.get_cumulative_reward()
            self.previous_cum_reward = current_reward
            
            # Get next images for next step
            self.get_current_image()
        else:
            # Penalty for invalid action
            reward = -1.0
        
        # Check if game should end
        if self.scorekeeper.remaining_time <= 0:
            finished = True
            # Add final score bonus/penalty
            final_score = self.scorekeeper.get_final_score()
            reward += final_score * 0.1  # Scale final score contribution
        
        # Update observation space
        self.get_observation_space()
        
        return self.observation_space, reward, finished, truncated, {}
    
    def _calculate_shaped_reward(self, action_idx):
        """
        Calculate immediate shaped reward based on action and game state
        Enhanced with occupation-based bonuses to incentivize learning occupation effects
        Provides much better learning signal than sparse score differences
        """
        reward = 0.0
        
        # Get CNN predictions for both sides (status indices only)
        left_status_idx = self.current_humanoid_probs_left[0]  # First value is status index
        right_status_idx = self.current_humanoid_probs_right[0]  # First value is status index
        left_occupation_idx = self.current_humanoid_probs_left[1]
        right_occupation_idx = self.current_humanoid_probs_right[1]
        
        # Map predictions to states for easier logic
        states = ['zombie', 'healthy', 'injured', 'corpse']
        left_predicted = states[left_status_idx]
        right_predicted = states[right_status_idx]
        
        # Use fixed confidence for now (can be enhanced later)
        left_confidence = 1.0
        right_confidence = 1.0
        
        # Base rewards for different actions on different entity types
        action_rewards = {
            'save_zombie': -2.0,    # Very bad - saving zombies
            'save_human': +1.5,     # Good - saving humans/injured
            'squish_zombie': +1.0,  # Good - eliminating threats
            'squish_human': -1.5,   # Bad - killing humans
            'skip_zombie': -0.5,    # Missed opportunity to eliminate threat
            'skip_human': -0.3,     # Missed opportunity to save
            'scram': 0.0           # Neutral base (context-dependent)
        }
        
        # Calculate action-specific rewards
        capacity_ratio = sum(self.scorekeeper.ambulance.values()) / self.scorekeeper.capacity
        zombie_ratio = self.scorekeeper.ambulance.get("zombie", 0) / max(1, sum(self.scorekeeper.ambulance.values()))
        time_ratio = self.scorekeeper.remaining_time / self.scorekeeper.shift_len
        
        if action_idx == 0:  # SKIP_BOTH
            # Penalty based on what we're skipping
            left_penalty = action_rewards.get(f'skip_{left_predicted}', -0.2) * left_confidence
            right_penalty = action_rewards.get(f'skip_{right_predicted}', -0.2) * right_confidence
            reward = (left_penalty + right_penalty) * 0.5
            
        elif action_idx == 1:  # SQUISH_LEFT
            if left_predicted in ['zombie', 'corpse']:
                reward = action_rewards['squish_zombie'] * left_confidence
            else:
                reward = action_rewards['squish_human'] * left_confidence
                
        elif action_idx == 2:  # SQUISH_RIGHT
            if right_predicted in ['zombie', 'corpse']:
                reward = action_rewards['squish_zombie'] * right_confidence
            else:
                reward = action_rewards['squish_human'] * right_confidence
                
        elif action_idx == 3:  # SAVE_LEFT
            if left_predicted == 'zombie':
                reward = action_rewards['save_zombie'] * left_confidence
            else:
                reward = action_rewards['save_human'] * left_confidence
                # Bonus for saving injured over healthy
                if left_predicted == 'injured':
                    reward += 0.3
                    
        elif action_idx == 4:  # SAVE_RIGHT
            if right_predicted == 'zombie':
                reward = action_rewards['save_zombie'] * right_confidence
            else:
                reward = action_rewards['save_human'] * right_confidence
                # Bonus for saving injured over healthy
                if right_predicted == 'injured':
                    reward += 0.3
                    
        elif action_idx == 5:  # SCRAM
            # SCRAM is good when capacity is full or zombie ratio is high
            if capacity_ratio > 0.8:
                reward = +1.0  # Good decision when near capacity
            elif zombie_ratio > 0.3:
                reward = +0.8  # Good decision when zombies are spreading
            else:
                reward = -0.5  # Wasteful if not necessary
                
        # Time pressure bonus/penalty
        if time_ratio < 0.2:  # Less than 20% time remaining
            if action_idx == 5:  # SCRAM
                reward += 0.5  # Bonus for scramming when time is low
            elif action_idx in [3, 4]:  # SAVE actions
                reward -= 0.3  # Small penalty for risky saves when time is critical
                
        # Capacity management bonus
        if capacity_ratio > 0.9 and action_idx != 5:
            reward -= 0.5  # Penalty for not scramming when at capacity
            
                # Zombie management bonus
        if zombie_ratio > 0.4:
            if action_idx == 5:  # SCRAM
                reward += 0.7  # Big bonus for scramming when zombie infection is high
            elif action_idx in [1, 2] and (left_predicted == 'zombie' or right_predicted == 'zombie'):
                reward += 0.3  # Bonus for squishing zombies when infection is spreading
                 
        return reward
    
    def _execute_action_with_validation(self, action_idx):
        """
        Execute action with proper validation and clear error reporting
        Returns (success: bool, failure_reason: str)
        """
        # Pre-action validation
        if self.scorekeeper.remaining_time <= 0:
            return False, "time_out"
            
        if action_idx < 0 or action_idx > 5:
            return False, "invalid_action"
        
        try:
            if action_idx == 0:  # SKIP_BOTH
                self.scorekeeper.skip_both(self.current_image_left, self.current_image_right)
                return True, "success"
                
            elif action_idx == 1:  # SQUISH_LEFT
                self.scorekeeper.squish(self.current_image_left)
                return True, "success"
                
            elif action_idx == 2:  # SQUISH_RIGHT
                self.scorekeeper.squish(self.current_image_right)
                return True, "success"
                
            elif action_idx == 3:  # SAVE_LEFT
                if self.scorekeeper.at_capacity():
                    return False, "capacity_full"
                self.scorekeeper.save(self.current_image_left)
                # Update vehicle storage tracking
                self._update_vehicle_storage(self.current_humanoid_probs_left)
                return True, "success"
                
            elif action_idx == 4:  # SAVE_RIGHT
                if self.scorekeeper.at_capacity():
                    return False, "capacity_full"
                self.scorekeeper.save(self.current_image_right)
                # Update vehicle storage tracking
                self._update_vehicle_storage(self.current_humanoid_probs_right)
                return True, "success"
                
            elif action_idx == 5:  # SCRAM
                self.scorekeeper.scram(self.current_image_left, self.current_image_right)
                # Clear vehicle storage when scramming
                self.observation_space["vehicle_storage_class_probs"] = np.zeros(
                    (self.environment_params['car_capacity'], self.environment_params['num_classes'])
                )
                return True, "success"
                
        except Exception as e:
            print(f"Action execution error: {e}")
            return False, "execution_error"
            
        return False, "unknown_error"
    
    def _update_vehicle_storage(self, humanoid_probs):
        """Update vehicle storage tracking when saving someone (store status as one-hot)"""
        current_capacity = self.scorekeeper.get_current_capacity()
        if current_capacity > 0 and current_capacity <= len(self.observation_space["vehicle_storage_class_probs"]):
            # Only store status as one-hot vector
            status_idx = int(humanoid_probs[0])
            one_hot = np.zeros(self.environment_params['num_classes'])
            one_hot[status_idx] = 1.0
            self.observation_space["vehicle_storage_class_probs"][current_capacity-1] = one_hot 