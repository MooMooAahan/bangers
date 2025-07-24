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
    def __init__(self, root, w, h, data_parser, scorekeeper, classifier_model_file=os.path.join('models', 'baseline.pth'), img_data_root='data', display=False, ):
        """
        initializes RL training interface
        
        dataparser : stores humanoid information needed to retreive humanoid images and rewards
        scorekeeper : keeps track of actions being done on humanoids, score, and is needed for reward calculations
        classifier_model_file : backbone model weights used in RL observation state
        """
        self.img_data_root = img_data_root
        self.data_parser = data_parser
        self.scorekeeper = scorekeeper
        self.display = display

        self.environment_params = {
            "car_capacity" : self.scorekeeper.capacity,
            "num_classes" : len(Humanoid.get_all_states()),
            "num_actions" : self.scorekeeper.actions,
        }
        self.observation_space = {"variables": np.zeros(3),
                                  "vehicle_storage_class_probs" : np.zeros((self.environment_params['car_capacity'],self.environment_params['num_classes'])),
                                    "humanoid_class_probs":np.zeros(self.environment_params['num_classes']),
                                    "doable_actions":np.ones(self.environment_params['num_actions'],np.int64),
                                    }

        self.action_space = spaces.Discrete(self.environment_params['num_actions'],)
        
        self.predictor = Predictor(classifier_model_file)
        
        #helper variables
        self.reset()

        if self.display:
            self.canvas = tk.Canvas(root, width=math.floor(0.2 * w), height=math.floor(0.1 * h))
            self.canvas.place(x=math.floor(0.75 * w), y=math.floor(0.75 * h))
            self.label = tk.Label(self.canvas, text="Simon says...", font=("Arial", 20))
            self.label.pack(side=tk.TOP)

            self.suggestion = tk.Label(self.canvas, text=self.text, font=("Arial", 20))
            self.suggestion.pack(side=tk.TOP)
            
    def reset(self):
        """
        resets game for a new episode to run.
        returns observation space
        """
        self.observation_space = {"variables": np.zeros(3),
                                  "vehicle_storage_class_probs" : np.zeros((self.environment_params['car_capacity'],self.environment_params['num_classes'])),
                                    "humanoid_class_probs":np.zeros(self.environment_params['num_classes']),
                                    "doable_actions":np.ones(self.environment_params['num_actions'],np.int64),
                                    }
        self.previous_cum_reward = 0
        self.data_parser.reset()
        self.scorekeeper.reset()
        self.get_humanoid()
        return self.observation_space
    
    def get_humanoid(self): ## TODO: this is currently hardcoded to left side
        """
        gets a random humanoid from the dataparser
        """
        self.humanoid = self.data_parser.get_random(side = 'left') ## TODO: this is currently hardcoded to left side
        img_ = Image.open(os.path.join(self.img_data_root, self.humanoid.fp))
        self.humanoid_probs = self.predictor.get_probs(img_) 
    
    def get_observation_space(self):
        """
        updates the observation space
        """
        self.observation_space['variables'] = np.array([self.scorekeeper.remaining_time, 
                                                        self.previous_cum_reward,
                                                        sum(self.scorekeeper.ambulance.values()),
                                                        ])
        self.observation_space["doable_actions"] = self.scorekeeper.available_action_space()
        self.observation_space["humanoid_class_probs"] = self.humanoid_probs
        
    def step(self, action_idx):
        """
        Acts on the environment and returns the observation state, reward, etc.
        
        action_idx : the index of the action being taken
        """
        action = ScoreKeeper.get_action_string(action_idx)
        
        reward=0
        finished = False # is game over
        
        action_executed = self.scorekeeper.map_do_action(action_idx, self.humanoid)
        
        if action_executed:
            reward = self.scorekeeper.get_cumulative_reward() - self.previous_cum_reward
            self.previous_cum_reward = self.scorekeeper.get_cumulative_reward()
            if action == "save":
                self.observation_space["vehicle_storage_class_probs"][self.scorekeeper.get_current_capacity()-1] = self.humanoid_probs
            elif action == "scram":
                self.observation_space["vehicle_storage_class_probs"] = np.zeros((self.environment_params['car_capacity'],self.environment_params['num_classes']))
            self.get_humanoid()
        else:
            reward-=0.5
        
        if self.scorekeeper.remaining_time <= 0:
            finished=True
            self.reset()
            
        self.get_observation_space()
        return self.observation_space, reward, finished, False, {} #self.observation_space, reward, finished, False, {} 