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
from endpoints.heuristic_interface import HeuristicInterface, Predictor
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
            "vehicle_storage_summary": np.zeros(4),  # [zombie_ratio, healthy_ratio, injured_ratio, corpse_ratio]
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
            from ui_elements.theme import UPGRADE_FONT
            self.label = tk.Label(self.canvas, text="RL Agent Training...", font=UPGRADE_FONT)
            self.label.pack(side=tk.TOP)

            self.suggestion = tk.Label(self.canvas, text="", font=UPGRADE_FONT)
            self.suggestion.pack(side=tk.TOP)
            
    def reset(self):
        """
        resets game for a new episode to run.
        returns observation space
        """
        # Reset observation space - simplified with status + occupation
        self.observation_space = {
            "variables": np.zeros(4),  # [time_ratio, reward_scaled, capacity_ratio, zombie_ratio]
            "vehicle_storage_summary": np.zeros(4),  # [zombie_ratio, healthy_ratio, injured_ratio, corpse_ratio]
            "humanoid_class_probs": np.zeros(4),  # Simplified: 4 values (2 per side: status + occupation)
            "doable_actions": np.ones(self.environment_params['num_actions'], dtype=np.int64),
        }
        
        self.previous_cum_reward = 0
        self.score_before_action = 0  # Initialize score tracking
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
            
            # 🎯 GROUND TRUTH: Use actual metadata instead of CNN predictions
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
            
            left_status_idx, left_occupation_idx = get_ground_truth_indices(self.current_image_left)
            right_status_idx, right_occupation_idx = get_ground_truth_indices(self.current_image_right)
            
            self.current_humanoid_probs_left = np.array([left_status_idx, left_occupation_idx])
            self.current_humanoid_probs_right = np.array([right_status_idx, right_occupation_idx])
            
            # For backward compatibility, create one-hot status probabilities from ground truth
            self.current_status_probs_left = np.zeros(self.environment_params['num_classes'])
            self.current_status_probs_left[left_status_idx] = 1.0
            self.current_status_probs_right = np.zeros(self.environment_params['num_classes']) 
            self.current_status_probs_right[right_status_idx] = 1.0
            
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
        
        # Update vehicle storage - sum across all slots for BaseModel (expects 4 dims, not 40)
        total_in_ambulance = sum(self.scorekeeper.ambulance.values())
        if total_in_ambulance > 0:
            # Aggregate ambulance composition into 4 values total (not per slot)
            vehicle_summary = np.array([
                self.scorekeeper.ambulance.get("zombie", 0) / max(1, total_in_ambulance),
                self.scorekeeper.ambulance.get("healthy", 0) / max(1, total_in_ambulance), 
                self.scorekeeper.ambulance.get("injured", 0) / max(1, total_in_ambulance),
                0  # corpse placeholder
            ])
        else:
            vehicle_summary = np.zeros(4)
        
        # Store as single summary vector (BaseModel will use this directly)
        self.observation_space["vehicle_storage_summary"] = vehicle_summary
        
        # Remove old format to prevent dimension confusion
        if "vehicle_storage_class_probs" in self.observation_space:
            del self.observation_space["vehicle_storage_class_probs"]
        
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
        
        # Capture score before action for reward calculation
        self.score_before_action = self.scorekeeper.get_final_score(route_complete=True)
        
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
            # Calculate reward based on actual score difference
            reward = self._calculate_actual_score_reward(action_idx, self.score_before_action)
            
            # Update cumulative tracking for consistency
            current_reward = self.scorekeeper.get_cumulative_reward()
            self.previous_cum_reward = current_reward
            
            # Get next images for next step
            self.get_current_image()
        else:
            # Penalty for invalid action (scaled to match new reward system)
            reward = -5.0  # Larger penalty since we're using actual scores now
        
        # Check if game should end
        if self.scorekeeper.remaining_time <= 0:
            finished = True
            # Add final score bonus/penalty
            final_score = self.scorekeeper.get_final_score()
            reward += final_score * 0.1  # Scale final score contribution
        
        # Update observation space
        self.get_observation_space()
        
        return self.observation_space, reward, finished, truncated, {}
    
    def _calculate_actual_score_reward(self, action_idx, score_before_action):
        """
        Calculate reward based on actual game score differences
        This ensures RL agent optimizes for the real 450+ scoring system
        """
        # Calculate current score after action execution
        score_after_action = self.scorekeeper.get_final_score(route_complete=True)
        
        # Use score difference as reward (scaled for RL stability)
        raw_reward = score_after_action - score_before_action
        
        # Light scaling to prevent gradient explosion while preserving score relationships
        # This maintains the relative importance of different actions
        scaled_reward = raw_reward / 10.0  # Scale by 10 to keep rewards in reasonable range
        
        # Add small time pressure component (preserves original time management incentive)
        time_ratio = self.scorekeeper.remaining_time / self.scorekeeper.shift_len
        if time_ratio < 0.1:  # Less than 10% time remaining
            scaled_reward += 1.0  # Small bonus for any action when time is critical
        
        # Debug output to track actual score progression
        action_names = ["SKIP_BOTH", "SQUISH_LEFT", "SQUISH_RIGHT", "SAVE_LEFT", "SAVE_RIGHT", "SCRAM"]
        action_name = action_names[action_idx] if 0 <= action_idx < len(action_names) else f"UNKNOWN_{action_idx}"
        
        # Only print every 10 actions to avoid spam
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        if self._step_count % 10 == 0:
            print(f"[SCORE] Action {action_name}: {score_before_action:.1f} → {score_after_action:.1f} (Δ{raw_reward:+.1f}, Reward: {scaled_reward:+.2f})")
            
        return scaled_reward
    
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
                self.observation_space["vehicle_storage_summary"] = np.zeros(4)
                return True, "success"
                
        except Exception as e:
            print(f"Action execution error: {e}")
            return False, "execution_error"
            
        return False, "unknown_error"
    
    def _update_vehicle_storage(self, humanoid_probs):
        """Update vehicle storage tracking when saving someone (handled by get_observation_space)"""
        # Vehicle storage summary is automatically updated in get_observation_space()
        # based on actual scorekeeper.ambulance contents, so no manual update needed
        pass 