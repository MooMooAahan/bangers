from gameplay.upgrades_manager import UpgradeManager
from gameplay.enums import ActionCost, ActionState
import pandas as pd
import random


MAP_ACTION_STR_TO_INT = {s.value:i for i,s in enumerate(ActionState)}
MAP_ACTION_INT_TO_STR = [s.value for s in ActionState]

"""
scoring system's global variables
"""
# Score for each type of person at the end of the game
SCORE_HEALTHY = 10
SCORE_INJURED = 5
SCORE_ZOMBIE = -20
SCORE_KILLED = -10
"""
timing system's global variables
"""

TIME_PENALTY_FOR_ZOMBIE = -15 #penalty for saving a zombie
TIME_BONUS_FOR_SAVING_HUMAN = +15 #time bonus for saving a human

class ScoreKeeper(object):
    def __init__(self, shift_len, capacity):
        
        self.shift_len = int(shift_len)  # minutes
        self.capacity = capacity
        
        self.actions = 4
        
        self.logger = []
        self.all_logs = []
        
        self.correct_saves = 0
        self.false_saves = 0
        
        self.reset()
        
        self.ambulance_time_adjustment = 0
        self.upgrade_manager = UpgradeManager(self)
        
    def reset(self):
        """
        resets scorekeeper on new environment
        """
        if hasattr(self, "upgrade_manager"):
            self.upgrade_manager.reset()

        self.ambulance = {
            "zombie": 0,
            "injured": 0,
            "healthy": 0
        }
        self.scorekeeper = {
            "killed": 0,
            "saved": 0,
        }
        self.remaining_time = int(self.shift_len)  # minutes
        
        self.all_logs.append(self.logger)
        self.logger = []
    
    def log(self, humanoid, action):
        """
        logs current action taken against a humanoid
        
        humanoid : the humanoid presented
        action : the action taken
        """
        self.logger.append({"humanoid_class":humanoid.state,
                            "humanoid_fp":humanoid.fp,
                            "action":action,
                            "remaining_time":self.remaining_time,
                            "capacity":self.get_current_capacity(),
                            })
        
    def save_log(self,):
        """
        Saves a single log.csv file containing the actions that were taken,and the humanoids presented at the time. 
        Note: will overwrite previous logs
        """
        if len(self.logger) > 0:
            self.all_logs.append(self.logger)
        logs = []
        for i, log in enumerate(self.all_logs):
            log = pd.DataFrame(log)
            log['local_run_id'] = i
            logs.append(log)
        logs = pd.DataFrame(logs)
        logs.to_csv('log.csv')

    def save(self, humanoid):
        """
        saves the humanoid
        updates scorekeeper
        """
        self.log(humanoid, 'save')
        
        # self.remaining_time -= ActionCost.SAVE.value
        
        time_bonus = 0
        
        if humanoid.is_zombie():
            self.ambulance["zombie"] += 1
            self.false_saves += 1
            time_bonus = TIME_PENALTY_FOR_ZOMBIE # penalty for saving zombie is removing 10 minutes
        elif humanoid.is_injured():
            self.correct_saves += 1
            self.ambulance["injured"] += 1
            if random.random() < 0.8:
                time_bonus = TIME_BONUS_FOR_SAVING_HUMAN # make them have a 30 min bonus  
        elif humanoid.is_healthy():
            self.correct_saves += 1
            self.ambulance["healthy"] += 1
            if random.random() < 0.8:
                time_bonus = TIME_BONUS_FOR_SAVING_HUMAN # make them have a 30 min bonus 
        
        self.ambulance_time_adjustment += time_bonus
        print(f"[DEBUG] Time adjustment: adding {time_bonus} minutes to remaining time, time remaining {self.remaining_time}")
        

    def squish(self, humanoid):
        """
        squishes the humanoid
        updates scorekeeper
        """
        self.log(humanoid, 'squish')
        
        self.remaining_time -= ActionCost.SQUISH.value
        if not (humanoid.is_zombie() or humanoid.is_corpse()):
            self.scorekeeper["killed"] += 1

    def skip(self, humanoid):
        """
        skips the humanoid
        updates scorekeeper
        """
        self.log(humanoid, 'skip')
        
        self.remaining_time -= ActionCost.SKIP.value
        if humanoid.is_injured():
            self.scorekeeper["killed"] += 1

    def scram(self, humanoid = None):
        """
        scrams
        updates scorekeeper
        """
        if humanoid:
            self.log(humanoid, 'scram')
        
        # self.remaining_time -= ActionCost.SCRAM.value
        
        if self.ambulance["zombie"] > 0:
            self.scorekeeper["killed"] += self.ambulance["injured"] + self.ambulance["healthy"]
        else:
            self.scorekeeper["saved"] += self.ambulance["injured"] + self.ambulance["healthy"]

        self.remaining_time += self.ambulance_time_adjustment
        self.ambulance_time_adjustment = 0
       
        if hasattr(self, "upgrade_manager"):
            num_humans = self.ambulance["healthy"] + self.ambulance["injured"]
            earnings = num_humans * 10
            self.upgrade_manager.earn_money(earnings)
            print(f"[SCRAM] Earned ${earnings} for {num_humans} humans.")
        
        self.ambulance["zombie"] = 0
        self.ambulance["injured"] = 0
        self.ambulance["healthy"] = 0
    
    def available_action_space(self):
        """
        returns available action space as a list of bools
        """
        action_dict = {s.value:True for s in ActionState}
        if self.remaining_time <= 0:
            action_dict['save'] = False
            action_dict['squish'] = False
            action_dict['skip'] = False
        if self.at_capacity():
            action_dict['save'] = False
        return [action_dict[s.value] for s in ActionState]
        
    # do_action or return false if not possible
    def map_do_action(self, idx, humanoid):
        """
        does an action on a humanoid. Intended for RL use.
        
        idx : the action index 
        """
        if idx == 0:
            if self.remaining_time <= 0 or self.at_capacity():
                return False
            self.save(humanoid)
        elif idx == 1:
            if self.remaining_time <= 0:
                return False
            self.squish(humanoid)
        elif idx == 2:
            if self.remaining_time <= 0:
                return False
            self.skip(humanoid)
        elif idx == 3:
            self.scram(humanoid)
        else:
            raise ValueError("action index range exceeded")
        return True
        
    def get_cumulative_reward(self):
        """
        returns cumulative reward (current score)
        Note: the score can be denoted as anything, not set in stone
        """
        killed = self.scorekeeper["killed"]
        saved = self.scorekeeper["saved"] 
        if self.ambulance["zombie"] > 0:
            killed += self.ambulance["injured"] + self.ambulance["healthy"]
        else:
            saved += self.ambulance["injured"] + self.ambulance["healthy"]
        return saved - killed

    def get_current_capacity(self):
        return sum(self.ambulance.values())

    def at_capacity(self):
        return sum(self.ambulance.values()) >= self.capacity

    def get_score(self):
        self.scram()
        return self.scorekeeper
    
    def get_final_score(self):
        """
        Calculate the final score based on saved/killed and also on time remaining
        """
        score = 0
        score += self.ambulance["healthy"] * SCORE_HEALTHY
        score += self.ambulance["injured"] * SCORE_INJURED
        score += self.ambulance["zombie"] * SCORE_ZOMBIE
        score += self.scorekeeper["killed"] * SCORE_KILLED
        score += self.remaining_time
        return score
    
    def get_accuracy(self):
        """
        Accuracy = correctly saved humans / all human decisions (saves + bad saves + killed humans)
        """
        true_positives = self.correct_saves  # you saved a real human
        false_positives = self.false_saves   # you saved a zombie
        false_negatives = self.scorekeeper["killed"] # you killed a human
        
        total = true_positives + false_positives + false_negatives
        return true_positives / total if total > 0 else 0
    
    @staticmethod
    def get_action_idx(class_string):
        return MAP_ACTION_STR_TO_INT[class_string]
    
    @staticmethod
    def get_action_string(class_idx):
        return MAP_ACTION_INT_TO_STR[class_idx]
    
    @staticmethod
    def get_all_actions():
        return MAP_ACTION_INT_TO_STR
