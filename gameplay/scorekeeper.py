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
SCORE_HEALTHY = 20
SCORE_INJURED = 10
SCORE_ZOMBIE = -15
SCORE_KILLED = -10
SCORE_SAVED = 20
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
        self.ambulance_people = {}
        self.scorekeeper = {
            "killed": 0,
            "saved": 0,
            "zombie_cured": 0,
            "human_infected": 0,
            "zombie_killed": 0,
            "human_killed": 0
        }
        self.remaining_time = int(self.shift_len)  # minutes
        
        self.all_logs.append(self.logger)
        self.logger = []
        
    
    def log(self, image, action):
        """
        logs current action taken against a humanoid
        
        humanoid : the humanoid presented
        action : the action taken
        """

        for humanoid in image.humanoids:
            if humanoid is None:
                continue
            self.logger.append({"humanoid_class":humanoid.state,
                    "humanoid_fp":humanoid.fp,
                    "action":action,
                    "remaining_time":self.remaining_time,
                    "capacity":self.get_current_capacity(),
                    })
    
    def logScram(self, image_left, image_right, action = 'scram'):
        """
        version for two images aka scram

        logs current action taken against a humanoid
        
        humanoid : the humanoid presented
        action : the action taken
        """

        for humanoid in image_left.humanoids:
            if humanoid is None:
                continue
            self.logger.append({"humanoid_class":humanoid.state,
                    "humanoid_fp":humanoid.fp,
                    "action":action,
                    "remaining_time":self.remaining_time,
                    "capacity":self.get_current_capacity(),
                    })
        for humanoid in image_right.humanoids:
            if humanoid is None:
                continue
            self.logger.append({"humanoid_class":humanoid.state,
                    "humanoid_fp":humanoid.fp,
                    "action":action,
                    "remaining_time":self.remaining_time,
                    "capacity":self.get_current_capacity(),
                    })
        
    def save_log(self):
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

    def save(self, image):
        """
        saves the humanoid
        updates scorekeeper
        """
        self.log(image, 'save')
        self.remaining_time -= ActionCost.SAVE.value
        time_bonus = 0
        # No longer add to ambulance_people here; handled by save_side_from_scenario
        filename = image.datarow['Filename']
        class_val = image.datarow['Class']
        humanoid_count = image.datarow['HumanoidCount']
        status_val = image.datarow['Injured']
        print(f"[DEBUG] Fileaaname: {filename}, Class: {class_val}, Status: {status_val}, HumanoidCount: {humanoid_count}")

        if humanoid_count == 1:
            self.ambulance_people[filename] = {
                "class": class_val,
                "injured": status_val,
                "role": "Civilian",
                "original_status": class_val
            }
            print(f"[DEBUG] Ambulance contents updated: {self.ambulance_people}")
        elif humanoid_count == 2:
            classes = class_val.split('|')
            classes[1] = classes[1].capitalize()
            injuries = status_val.split('|')
            self.ambulance_people[filename + "_1"] = {
                "class": classes[0],
                "injured": injuries[0],
                "role": "Civilian",
                "original_status": classes[0]
            }
            self.ambulance_people[filename + "_2"] = {
                "class": classes[1],
                "injured": injuries[1],
                "role": "Civilian",
                "original_status": classes[1]
            }
        else:
            pass
        
        print(f"[DEBUG] Ambulance contents updated: {self.ambulance_people}")

        if humanoid_count == 1:
            if class_val == "Zombie":
                self.ambulance["zombie"] += 1
                self.false_saves += 1
                time_bonus = TIME_PENALTY_FOR_ZOMBIE
            elif class_val == "Default":
                if status_val == 'True':
                    self.correct_saves += 1
                    self.ambulance["injured"] += 1
                    if random.random() < 0.8:
                        time_bonus = TIME_BONUS_FOR_SAVING_HUMAN
                else:
                    self.ambulance["healthy"] += 1
                    self.correct_saves += 1
                    if random.random() < 0.8:
                        time_bonus = TIME_BONUS_FOR_SAVING_HUMAN
            else:
                pass
                """
        for doubled humanoids, lowk just bashed this out
                """
        elif humanoid_count == 2:
            if classes[0] == "Zombie":
                self.ambulance["zombie"] += 1
                self.false_saves += 1
                time_bonus = TIME_PENALTY_FOR_ZOMBIE
            elif classes[0] == "Default":
                if injuries[0] == 'True':
                    self.correct_saves += 1
                    self.ambulance["injured"] += 1
                    if random.random() < 0.8:
                        time_bonus = TIME_BONUS_FOR_SAVING_HUMAN
                else:
                    self.ambulance["healthy"] += 1
                    self.correct_saves += 1
                    if random.random() < 0.8:
                        time_bonus = TIME_BONUS_FOR_SAVING_HUMAN
            else:
                pass

            if classes[1] == "Zombie":
                self.ambulance["zombie"] += 1
                self.false_saves += 1
                time_bonus = TIME_PENALTY_FOR_ZOMBIE
            elif classes[1] == "Default":
                if injuries[1] == 'True':
                    self.correct_saves += 1
                    self.ambulance["injured"] += 1
                    if random.random() < 0.8:
                        time_bonus = TIME_BONUS_FOR_SAVING_HUMAN
                else:
                    self.ambulance["healthy"] += 1
                    self.correct_saves += 1
                    if random.random() < 0.8:
                        time_bonus = TIME_BONUS_FOR_SAVING_HUMAN
            else:
                pass
        else:
            pass

    """
    end of those annoying shenanigans
    """
        

    def squish(self, image):
        """
        squishes the humanoid
        updates scorekeeper
        """
        self.log(image, 'squish')

        filename = image.datarow['Filename']
        class_val = image.datarow['Class']
        humanoid_count = image.datarow['HumanoidCount']
        status_val = image.datarow['Injured']

        self.remaining_time -= ActionCost.SQUISH.value
        if humanoid_count == 1:
            if class_val == "Zombie":
                self.scorekeeper["zombie_killed"] += 1
            elif class_val == "Default":
                self.scorekeeper["human_killed"] += 1
            else:
                pass
        
        elif humanoid_count == 2:
            classes = class_val.split('|')
            classes[1] = classes[1].capitalize()
            injuries = status_val.split('|')
            if classes[0] == "Zombie":
                self.scorekeeper["zombie_killed"] += 1
            elif classes[0] == "Default":
                self.scorekeeper["human_killed"] += 1
            else:
                pass

            if classes[1] == "Zombie":
                self.scorekeeper["zombie_killed"] += 1
            elif classes[1] == "Default":
                self.scorekeeper["human_killed"] += 1
            else:
                pass
        else:
            pass

    def skip(self, image):
        """
        skips the humanoid
        updates scorekeeper
        """
        self.log(image, 'skip')
        
        self.remaining_time -= ActionCost.SKIP.value
        for humanoid in image.humanoids:
            if humanoid is None:
                continue
            if humanoid.is_injured():
                self.scorekeeper["killed"] += 1

    def skip_both(self, image_left, image_right):
        """Skips both humanoids but only deducts 15 minutes total."""
        # access image humanoids and log them 
        self.log(image_left, 'skip')
        self.log(image_right, 'skip')
        self.remaining_time -= ActionCost.SKIP.value
        for humanoid in image_left.humanoids:
            if humanoid is None:
                continue
            if humanoid.is_injured():
                self.scorekeeper["killed"] += 1
        for humanoid in image_right.humanoids:
            if humanoid is None:
                continue
            if humanoid.is_injured():
                self.scorekeeper["killed"] += 1

    def inspect(self, image, cost=None):
        """Logs an inspect action and deducts inspect cost."""
        for humanoid in image.humanoids:
            if humanoid is None:
                continue
            self.log(image, 'inspect')

        if cost is None:
            cost = ActionCost.INSPECT.value
        self.remaining_time -= cost

    #TODO: fix this to be able to scram images and not humanoids + other number tweaks
    def scram(self, image_left, image_right, time_cost=None):
        """
        scrams
        updates scorekeeper
        """
        self.logScram(image_left, image_right, 'scram')
        if time_cost is not None:
            self.remaining_time -= time_cost
        else:
            self.remaining_time -= ActionCost.SCRAM.value
        
        # Count zombies as killed, humans as saved
        self.scorekeeper["killed"] += self.ambulance["zombie"]
        self.scorekeeper["saved"] += self.ambulance["injured"] + self.ambulance["healthy"]

        if hasattr(self, "upgrade_manager"):
            num_humans = self.ambulance["healthy"] + self.ambulance["injured"]
            earnings = num_humans * 10
            self.upgrade_manager.earn_money(earnings)
            print(f"[SCRAM] Earned ${earnings} for {num_humans} humans.")
        
        self.ambulance["zombie"] = 0
        self.ambulance["injured"] = 0
        self.ambulance["healthy"] = 0
        self.ambulance_people.clear()
        print(f"[DEBUG] Ambulance cleared: {self.ambulance_people}")
    
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
    
    def get_final_score(self, route_complete=True):
        """
        Calculate the final score based on saved/killed and also on time remaining
        """
        score = 0
        score += self.ambulance["healthy"] * SCORE_HEALTHY
        score += self.ambulance["injured"] * SCORE_INJURED
        score += self.ambulance["zombie"] * SCORE_ZOMBIE
        score += self.scorekeeper["killed"] * SCORE_KILLED
        score += self.scorekeeper["saved"] * SCORE_SAVED
        score += self.scorekeeper["zombie_cured"] * 25
        score += self.scorekeeper["human_infected"] * -15
        score += self.scorekeeper["zombie_killed"] * 10
        score += self.scorekeeper["human_killed"] * -10
        score += self.remaining_time
        if not route_complete:
            score -= 500
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

    def process_zombie_infections(self):
        """
        Process zombie infections at the start of each turn.
        Each zombie has a 5% chance to turn each human into a zombie, reduced by militants.
        Each healthy militant reduces infection chance by 10%, each injured militant by 5%.
        """
        # Count zombies and humans
        zombie_count = self.ambulance["zombie"]
        human_count = self.ambulance["injured"] + self.ambulance["healthy"]

        # Count militants
        healthy_militants = 0
        injured_militants = 0
        for person in self.ambulance_people.values():
            if person['role'] == 'Militant':
                if person['injured'] == 'True':
                    injured_militants += 1
                else:
                    healthy_militants += 1

        if zombie_count == 0 or human_count == 0:
            return []  # No infections possible

        infected_humanoids = []

        # Check each humanoid in the ambulance
        for humanoid_id, humanoid_data in list(self.ambulance_people.items()):
            if humanoid_data["class"] == "Default":
                # Each zombie has a 5% chance to infect this human
                infection_chance = zombie_count * 0.05
                # Subtract militant protection
                infection_chance -= healthy_militants * 0.10
                infection_chance -= injured_militants * 0.05
                infection_chance = max(0, infection_chance)  # Don't allow negative chance
                if random.random() < infection_chance:
                    # Turn this human into a zombie
                    humanoid_data["class"] = "Zombie"
                    humanoid_data["role"] = "blank"

                    # Update ambulance counts
                    if humanoid_data.get("status") == "True":
                        self.ambulance["injured"] -= 1
                    else:
                        self.ambulance["healthy"] -= 1
                    self.ambulance["zombie"] += 1
                    self.scorekeeper["human_infected"] += 1

                    infected_humanoids.append(humanoid_id)

        return infected_humanoids

    def process_zombie_cures(self):
        """
        Process zombie cures at the start of each turn.
        Each healthy doctor adds 5% cure chance, each injured doctor adds 2.5% cure chance.
        Each zombie is checked independently.
        """
        # Count doctors
        cure_chance = 0.0
        for person in self.ambulance_people.values():
            if person['role'] == 'Doctor':
                if person['injured'] == 'True':
                    cure_chance += 0.025
                else:
                    cure_chance += 0.05
        if cure_chance == 0:
            return []  # No cures possible
        cured_humanoids = []
        for humanoid_id, humanoid_data in list(self.ambulance_people.items()):
            if humanoid_data["class"] == "Zombie":
                if random.random() < cure_chance:
                    # Cure this zombie
                    humanoid_data["class"] = "Default"
                    humanoid_data["role"] = "Civilian"
                    # Keep status (injured/healthy) the same
                    humanoid_data["original_status"] = "cured_zombie"
                    self.scorekeeper["zombie_cured"] += 1
                    cured_humanoids.append(humanoid_id)
        return cured_humanoids





    def save_side_from_scenario(self, side, scenario_humanoid_attributes):
        """
        Save all nonblank people from scenario_humanoid_attributes for the given side ('left' or 'right').
        """
        for i in range(1, 4):
            key = f"{side}_humanoid{i}"
            attrs = scenario_humanoid_attributes.get(key, {})
            if attrs and attrs.get('type', '').strip():
                humanoid_id = f"humanoid{len(self.ambulance_people) + 1}"
                self.ambulance_people[humanoid_id] = {
                    "class": attrs['type'],
                    "status": attrs['status'],
                    "role": attrs['role'],
                    "original_status": attrs['status']

                }
            if attrs['type'] == 'zombie':
                self.false_saves += 1
            