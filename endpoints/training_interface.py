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

from gym import Env, spaces
from endpoints.data_parser import DataParser


class TrainInterface(Env):
    def __init__(self, root=None, w=800, h=600, data_parser=None, scorekeeper=None, 
                 classifier_model_file=os.path.join('models', 'baseline.pth'), 
                 img_data_root='data', display=False):
        """
        initializes RL training interface
        
        dataparser : stores humanoid information needed to retreive humanoid images and rewards
        scorekeeper : keeps track of actions being done on humanoids, score, and is needed for reward calculations
        classifier_model_file : backbone model weights used in RL observation state
        """
        self.img_data_root = img_data_root
        self.data_parser = data_parser if data_parser else DataParser(img_data_root)
        self.scorekeeper = scorekeeper if scorekeeper else ScoreKeeper(shift_len=480, capacity=10)
        self.display = display

        self.environment_params = {
            "car_capacity" : self.scorekeeper.capacity,
            "num_classes" : len(Humanoid.get_all_states()),
            "num_actions" : 6,  # Updated: SKIP_BOTH, SQUISH_LEFT, SQUISH_RIGHT, SAVE_LEFT, SAVE_RIGHT, SCRAM
        }
        
        # Initialize observation space structure
        self.observation_space = {
            "variables": np.zeros(3),
            "vehicle_storage_class_probs": np.zeros((self.environment_params['car_capacity'], self.environment_params['num_classes'])),
            "humanoid_class_probs": np.zeros(self.environment_params['num_classes'] * 2),  # Both left and right
            "doable_actions": np.ones(self.environment_params['num_actions'], dtype=np.int64),
        }

        self.action_space = spaces.Discrete(self.environment_params['num_actions'])
        
        # Initialize CNN predictor for observations
        self.predictor = Predictor(classes=self.environment_params['num_classes'], 
                                 model_file=classifier_model_file)
        
        # Initialize state
        self.current_image_left = None
        self.current_image_right = None
        self.current_humanoid_probs_left = np.ones(self.environment_params['num_classes']) / self.environment_params['num_classes']
        self.current_humanoid_probs_right = np.ones(self.environment_params['num_classes']) / self.environment_params['num_classes']
        
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
        # Reset observation space
        self.observation_space = {
            "variables": np.zeros(4),  # Updated: time, reward, capacity, zombie_count
            "vehicle_storage_class_probs": np.zeros((self.environment_params['car_capacity'], self.environment_params['num_classes'])),
            "humanoid_class_probs": np.zeros(self.environment_params['num_classes'] * 2),  # Both left and right
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
            
            pil_img_left = Image.open(img_path_left)
            pil_img_right = Image.open(img_path_right)
            
            self.current_humanoid_probs_left = self.predictor.get_probs(pil_img_left)
            self.current_humanoid_probs_right = self.predictor.get_probs(pil_img_right)
            
        except Exception as e:
            print(f"Error loading images: {e}")
            # Fallback to random probabilities
            self.current_humanoid_probs_left = np.ones(self.environment_params['num_classes']) / self.environment_params['num_classes']
            self.current_humanoid_probs_right = np.ones(self.environment_params['num_classes']) / self.environment_params['num_classes']
    
    def get_observation_space(self):
        """
        Updates the observation space with current game state including both left and right scenarios
        """
        self.observation_space['variables'] = np.array([
            self.scorekeeper.remaining_time, 
            self.previous_cum_reward,
            sum(self.scorekeeper.ambulance.values()),
        ])
        
        self.observation_space["doable_actions"] = np.array(self.scorekeeper.available_action_space(), dtype=np.int64)
        
        # Include both left and right humanoid probabilities
        # Concatenate left and right probs to give agent full scenario information
        combined_probs = np.concatenate([self.current_humanoid_probs_left, self.current_humanoid_probs_right])
        self.observation_space["humanoid_class_probs"] = combined_probs
        
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
        
        # Add strategic information: zombie count (for police effect awareness)
        zombie_count_normalized = min(self.scorekeeper.ambulance.get("zombie", 0) / self.scorekeeper.capacity, 1.0)
        self.observation_space['variables'] = np.array([
            self.scorekeeper.remaining_time / self.scorekeeper.shift_len,  # Normalize to [0,1]
            self.previous_cum_reward / 100.0,  # Scale reward
            sum(self.scorekeeper.ambulance.values()) / self.scorekeeper.capacity,  # Capacity ratio
            zombie_count_normalized,  # Zombie ratio (important for police effect)
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
        
        # Execute action based on new action space
        action_executed = False
        
        try:
            if action_idx == 0:  # SKIP_BOTH
                if not (self.scorekeeper.remaining_time <= 0):
                    self.scorekeeper.skip_both(self.current_image_left, self.current_image_right)
                    action_executed = True
                    
            elif action_idx == 1:  # SQUISH_LEFT
                if not (self.scorekeeper.remaining_time <= 0):
                    self.scorekeeper.squish(self.current_image_left)
                    action_executed = True
                    
            elif action_idx == 2:  # SQUISH_RIGHT
                if not (self.scorekeeper.remaining_time <= 0):
                    self.scorekeeper.squish(self.current_image_right)
                    action_executed = True
                    
            elif action_idx == 3:  # SAVE_LEFT
                if not (self.scorekeeper.remaining_time <= 0 or self.scorekeeper.at_capacity()):
                    self.scorekeeper.save(self.current_image_left)
                    action_executed = True
                    # Update vehicle storage when saving
                    current_capacity = self.scorekeeper.get_current_capacity()
                    if current_capacity > 0 and current_capacity <= len(self.observation_space["vehicle_storage_class_probs"]):
                        self.observation_space["vehicle_storage_class_probs"][current_capacity-1] = self.current_humanoid_probs_left
                        
            elif action_idx == 4:  # SAVE_RIGHT
                if not (self.scorekeeper.remaining_time <= 0 or self.scorekeeper.at_capacity()):
                    self.scorekeeper.save(self.current_image_right)
                    action_executed = True
                    # Update vehicle storage when saving
                    current_capacity = self.scorekeeper.get_current_capacity()
                    if current_capacity > 0 and current_capacity <= len(self.observation_space["vehicle_storage_class_probs"]):
                        self.observation_space["vehicle_storage_class_probs"][current_capacity-1] = self.current_humanoid_probs_right
                        
            elif action_idx == 5:  # SCRAM
                self.scorekeeper.scram(self.current_image_left, self.current_image_right)
                action_executed = True
                # Clear vehicle storage when scramming
                self.observation_space["vehicle_storage_class_probs"] = np.zeros((self.environment_params['car_capacity'], self.environment_params['num_classes']))
                
            else:
                print(f"Invalid action index: {action_idx}")
                action_executed = False
                
        except Exception as e:
            print(f"Error executing action {action_idx}: {e}")
            action_executed = False
        
        if action_executed:
            # Calculate reward based on score improvement
            current_reward = self.scorekeeper.get_cumulative_reward()
            reward = current_reward - self.previous_cum_reward
            self.previous_cum_reward = current_reward
            
            # Get next images for next step
            self.get_current_image()
        else:
            # Penalty for invalid action
            reward = -0.5
        
        # Check if game should end
        if self.scorekeeper.remaining_time <= 0:
            finished = True
            # Add final score bonus/penalty
            final_score = self.scorekeeper.get_final_score()
            reward += final_score * 0.1  # Scale final score contribution
        
        # Update observation space
        self.get_observation_space()
        
        return self.observation_space, reward, finished, truncated, {} 