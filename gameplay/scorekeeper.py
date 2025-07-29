from gameplay.upgrades_manager import UpgradeManager
from gameplay.enums import ActionCost, ActionState
import pandas as pd
import random
import os
from datetime import datetime
import sys


MAP_ACTION_STR_TO_INT = {s.value:i for i,s in enumerate(ActionState)}
MAP_ACTION_INT_TO_STR = [s.value for s in ActionState]

def _is_automated_mode():
    """Check if we're in training or heuristic mode to suppress UI popups"""
    import sys
    # Only suppress popups if running RL training or heuristic mode
    return any(mode in sys.argv for mode in ['-m', 'train', 'heuristic']) and not any(mode in sys.argv for mode in ['user', 'infer'])

def _safe_show_popup(title, message, popup_type='info', display=True):
    """Show popup only if not in automated mode"""
    if display and not _is_automated_mode():
        try:
            import tkinter.messagebox
            if popup_type == 'warning':
                tkinter.messagebox.showwarning(title, message)
            else:
                tkinter.messagebox.showinfo(title, message)
        except Exception as e:
            # print(f'[DEBUG] Could not show popup: {e}')
            pass

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
    def __init__(self, shift_len, capacity, display=True, log_path='log.csv'):
        
        self.shift_len = int(shift_len)  # minutes
        self.capacity = capacity
        self.log_path = log_path  # Allow custom log file path
        
        self.actions = 4
        
        self.logger = []
        self.all_logs = []
        
        self.correct_saves = 0
        self.false_saves = 0
        
        self.reset()
        
        self.ambulance_time_adjustment = 0
        self.upgrade_manager = UpgradeManager(self)
        self.inspected_state = {('left', None): False, ('right', None): False}  # (side, humanoid_fp) -> bool
        self.display = display
        # Track inspection state for current scenario
        self.inspected_left = False
        self.inspected_right = False
        
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
        
        # Reset inspection state for new scenario
        self.inspected_left = False
        self.inspected_right = False
        
    
    def log_consolidated(self, action, image_left=None, image_right=None, route_position=None, action_side=None):
        """Log a single consolidated entry for the current game state"""
        timestamp = datetime.now().isoformat()
        
        # Get humanoid classes for left and right
        left_classes = []
        right_classes = []
        if image_left and hasattr(image_left, 'humanoids'):
            left_classes = [h.state for h in image_left.humanoids if h is not None]
        if image_right and hasattr(image_right, 'humanoids'):
            right_classes = [h.state for h in image_right.humanoids if h is not None]
        
        # Combine classes as comma-separated list
        humanoid_class = f"({', '.join(left_classes)}, {', '.join(right_classes)})"
        
        # Get roles for left and right
        left_roles = []
        right_roles = []
        if image_left and hasattr(image_left, 'humanoids'):
            left_roles = [h.role for h in image_left.humanoids if h is not None]
        if image_right and hasattr(image_right, 'humanoids'):
            right_roles = [h.role for h in image_right.humanoids if h is not None]
        
        # Combine roles as comma-separated list
        role = f"({', '.join(left_roles)}, {', '.join(right_roles)})"
        
        # Get inspection state for both sides using the new variables
        # First letter: left side inspection (Y/N)
        # Second letter: right side inspection (Y/N)
        inspected = f"{'Y' if self.inspected_left else 'N'}{'Y' if self.inspected_right else 'N'}"
        
        # Get capacity info
        total_people = len(self.ambulance_people)
        capacity_info = f"({total_people},{self.capacity})"
        
        # Determine action_side if not provided
        if action_side is None:
            if action in ['skip_both', 'scram']:
                action_side = 'both'
            else:
                action_side = 'unknown'  # Will be set by calling function
        
        log_entry = {
            "timestamp": timestamp,
            "local_run_id": None,  # Will be filled in save_log
            "route_position": route_position,
            "humanoid_class": humanoid_class,
            "capacity": capacity_info,
            "remaining_time": self.remaining_time,
            "role": role,
            "inspected": inspected,
            "action": action,
            "action_side": action_side
        }
        
        self.logger.append(log_entry)

    def log_final_score(self, final_score, movement_count, max_movements, route_complete):
        """Log final score and game statistics to CSV"""
        timestamp = datetime.now().isoformat()
        
        # Create final score entry
        final_entry = {
            "timestamp": timestamp,
            "local_run_id": 0,  # Will be updated below
            "route_position": movement_count,
            "humanoid_class": "FINAL_SCORE",
            "capacity": f"({sum(self.ambulance.values())},{self.capacity})",
            "remaining_time": self.remaining_time,
            "role": f"SCORE:{final_score}",
            "inspected": f"ROUTE:{'COMPLETE' if route_complete else 'INCOMPLETE'}",
            "action": "game_end",
            "action_side": f"movements_{movement_count}_{max_movements}"
        }
        
        # Convert to DataFrame
        final_df = pd.DataFrame([final_entry])
        
        # Determine the current local_run_id by reading existing file
        log_path = self.log_path
        if os.path.exists(log_path):
            try:
                existing = pd.read_csv(log_path)
                if 'local_run_id' in existing.columns and not existing.empty:
                    current_run_id = existing['local_run_id'].max()
                    final_df['local_run_id'] = current_run_id
                else:
                    final_df['local_run_id'] = 0
            except Exception:
                final_df['local_run_id'] = 0
        else:
            final_df['local_run_id'] = 0
        
        # Append to existing CSV
        if os.path.exists(log_path):
            try:
                existing = pd.read_csv(log_path)
                combined = pd.concat([existing, final_df], ignore_index=True)
            except Exception:
                combined = final_df
        else:
            combined = final_df
        
        combined.to_csv(log_path, index=False)
        print(f"🏆 Final score {final_score} logged to {log_path}")

    def save_log(self, final=False):
        if not final:
            self.logger = []
            return
        
        # Prepare new log DataFrame
        new_log = pd.DataFrame(self.logger)
        if new_log.empty:
            new_log = pd.DataFrame([{
                "timestamp": None, "local_run_id": None, "route_position": None,
                "humanoid_class": None, "capacity": None, "remaining_time": None,
                "role": None, "inspected": None, "action": None, "action_side": None
            }])

        # Fill missing route_position with last valid or -1
        if 'route_position' in new_log.columns:
            if new_log['route_position'].notnull().any():
                last_pos = new_log['route_position'].dropna().iloc[-1]
            else:
                last_pos = -1
            new_log['route_position'] = new_log['route_position'].fillna(last_pos)

        # Determine the next local_run_id
        log_path = self.log_path
        if os.path.exists(log_path):
            try:
                existing = pd.read_csv(log_path)
                if 'local_run_id' in existing.columns and not existing.empty:
                    last_id = existing['local_run_id'].max()
                    next_id = last_id + 1
                else:
                    next_id = 0
            except Exception:
                next_id = 0
        else:
            next_id = 0
        new_log['local_run_id'] = next_id

        # Reorder columns
        col_order = [
            'timestamp', 'local_run_id', 'route_position', 'humanoid_class',
            'capacity', 'remaining_time', 'role', 'inspected', 'action', 'action_side'
        ]
        new_log = new_log[col_order]

        # Append to file
        if os.path.exists(log_path):
            try:
                existing = pd.read_csv(log_path)
                combined = pd.concat([existing, new_log], ignore_index=True)
            except Exception:
                combined = new_log
        else:
            combined = new_log
        combined.to_csv(log_path, index=False)
        
        # Reset logger for next round
        self.logger = []

    def save(self, image, route_position=None, side=None, image_left=None, image_right=None):
        """
        saves the humanoid
        updates scorekeeper
        """
        # Check if there's enough space for this humanoid
        humanoid_count = image.datarow['HumanoidCount']
        current_people = len(self.ambulance_people)
        if current_people + humanoid_count > self.capacity:
            # Not enough space, don't save
            return 0
        
        self.log_consolidated('save', image_left=image_left, image_right=image_right, route_position=route_position, action_side=side)
        self.remaining_time -= ActionCost.SAVE.value
        time_bonus = 0
        # No longer add to ambulance_people here; handled by save_side_from_scenario
        filename = image.datarow['Filename']
        class_val = image.datarow['Class']
        status_val = image.datarow['Injured']
        roles_val = image.datarow['Role']
        # print(f"[DEBUG] Fileaaname: {filename}, Class: {class_val}, Status: {status_val}, HumanoidCount: {humanoid_count}, Role: {roles_val}")

        if humanoid_count == 1:
            self.ambulance_people[filename] = {
                "class": class_val,
                "injured": status_val,
                "role": roles_val,  # Always preserve the original role
                "original_status": class_val
            }
            # print(f"[DEBUG] Ambulance contents updated: {self.ambulance_people}")
        elif humanoid_count == 2:
            classes = class_val.split('|')
            classes[1] = classes[1].capitalize()
            injuries = status_val.split('|')
            roles = roles_val.split('|')
            # Always preserve the original roles, regardless of state
            self.ambulance_people[filename + "_1"] = {
                "class": classes[0],
                "injured": injuries[0],
                "role": roles[0],  # Always preserve the original role
                "original_status": classes[0]
            }
            self.ambulance_people[filename + "_2"] = {
                "class": classes[1],
                "injured": injuries[1],
                "role": roles[1],  # Always preserve the original role
                "original_status": classes[1]
            }
        else:
            pass
        
        # print(f"[DEBUG] Ambulance contents updated: {self.ambulance_people}")

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
        This is the police effect stuff, when you pick em up they kill a zombie
        """
        if class_val == "Default" and roles_val == "Police":
                    # Police effect: if you pick up a police, kill 1 zombie in the ambulance
            zombie_removed = False
            for humanoid_id, person in list(self.ambulance_people.items()):
                if person.get('class') == 'Zombie':
                    del self.ambulance_people[humanoid_id]
                    zombie_removed = True
                    # print(f'[DEBUG] Police picked up: Removed zombie entry {humanoid_id} from ambulance_people.')
                    break
            if zombie_removed:
                if self.ambulance['zombie'] > 0:
                    self.ambulance['zombie'] -= 1
                    self.scorekeeper['zombie_killed'] += 1
#                     # print('[DEBUG] Police picked up: 1 zombie removed from ambulance.')
#                 # Show popup message
#                 try:
#                     import tkinter.messagebox
#                     tkinter.messagebox.showinfo('Police Action', 'The Police you picked up killed a zombie!')
#                 except Exception as e:
#                     # print(f'[DEBUG] Could not show popup: {e}')
#                     pass
                    # print('[DEBUG] Police picked up: 1 zombie removed from ambulance.')
                # Show popup message  
                _safe_show_popup('Police Action', 'The Police you picked up killed a zombie!', display=self.display)
            else:
                _safe_show_popup('Police Action', 'There were no zombies for your police to kill!', display=self.display)
        elif humanoid_count == 2:
            if roles[0] == "Police" and classes[0] == "Default":
                                # Police effect: if you pick up a police, kill 1 zombie in the ambulance
                zombie_removed = False
                for humanoid_id, person in list(self.ambulance_people.items()):
                    if person.get('class') == 'Zombie':
                        del self.ambulance_people[humanoid_id]
                        zombie_removed = True
                        # print(f'[DEBUG] Police picked up: Removed zombie entry {humanoid_id} from ambulance_people.')
                        break
                if zombie_removed:
                    if self.ambulance['zombie'] > 0:
                        self.ambulance['zombie'] -= 1
                        self.scorekeeper['zombie_killed'] += 1
                        # print('[DEBUG] Police picked up: 1 zombie removed from ambulance.')
                    # Show popup message
#                     try:
#                         import tkinter.messagebox
#                         tkinter.messagebox.showinfo('Police Action', 'The Police you picked up killed a zombie!')
#                     except Exception as e:
#                         # print(f'[DEBUG] Could not show popup: {e}')
#                         pass
                    _safe_show_popup('Police Action', 'The Police you picked up killed a zombie!', display=self.display)
                else:
                    _safe_show_popup('Police Action', 'There were no zombies for your police to kill!', display=self.display)    
            elif roles[1] == "Police" and classes[1] == "Default":
                zombie_removed = False
                for humanoid_id, person in list(self.ambulance_people.items()):
                    if person.get('class') == 'Zombie':
                        del self.ambulance_people[humanoid_id]
                        zombie_removed = True
                        # print(f'[DEBUG] Police picked up: Removed zombie entry {humanoid_id} from ambulance_people.')
                        break
                if zombie_removed:
                    if self.ambulance['zombie'] > 0:
                        self.ambulance['zombie'] -= 1
                        self.scorekeeper['zombie_killed'] += 1
                        # print('[DEBUG] Police picked up: 1 zombie removed from ambulance.')
                    # Show popup message
#                     try:
#                         import tkinter.messagebox
#                         tkinter.messagebox.showinfo('Police Action', 'The Police you picked up killed a zombie!')
#                     except Exception as e:
#                         # print(f'[DEBUG] Could not show popup: {e}')
#                         pass
                    _safe_show_popup('Police Action', 'The Police you picked up killed a zombie!', display=self.display)
                else:
                    _safe_show_popup('Police Action', 'There were no zombies for your police to kill!', display=self.display)  
            else:
                pass

        else: 
            pass
            
        
        """
        end of those annoying shenanigans, start of beaver functionality
        """

        if class_val == "Beaver":
            # Transform all injured and zombie people in the ambulance to healthy
            beaver_transformed = False
            # Update ambulance_people
            for humanoid_id, person in self.ambulance_people.items():
                if person["class"] == "Zombie":
                    person["class"] = "Default"
                    person["injured"] = "False"
                    person["role"] = "Civilian"
                    beaver_transformed = True
                elif person["class"] == "Default" and person["injured"] == "True":
                    person["injured"] = "False"
                    beaver_transformed = True
            # Count how many were previously injured and zombies
            num_injured = self.ambulance["injured"]
            num_zombie = self.ambulance["zombie"]
            # All become healthy
            self.ambulance["healthy"] += num_injured + num_zombie
            self.ambulance["injured"] = 0
            self.ambulance["zombie"] = 0
            # Show popup message
            if beaver_transformed:
                _safe_show_popup('Beaver Magic', 'The Transformational Beaver made everyone in your ambulance healthy!', display=self.display)
            else:
                _safe_show_popup('Beaver Magic', 'You encountered the Magical Beaver... but there was no one to save!', display=self.display)

    def squish(self, image, route_position=None, side=None, image_left=None, image_right=None):
        """
        squishes the humanoid
        updates scorekeeper
        """
        self.log_consolidated('squish', image_left=image_left, image_right=image_right, route_position=route_position, action_side=side)

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

    def skip(self, image, route_position=None, side=None, image_left=None, image_right=None):
        """
        skips the humanoid
        updates scorekeeper
        """
        self.log_consolidated('skip', image_left=image_left, image_right=image_right, route_position=route_position, action_side=side)
        
        self.remaining_time -= ActionCost.SKIP.value
        for humanoid in image.humanoids:
            if humanoid is None:
                continue
            if humanoid.is_injured():
                self.scorekeeper["killed"] += 1

    def skip_both(self, image_left, image_right, route_position=None):
        """Skips both humanoids but only deducts 15 minutes total."""
        # For skip both, neither side is chosen, so both get action 'skip'
        self.log_consolidated('skip_both', image_left=image_left, image_right=image_right, route_position=route_position, action_side='both')
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

    def inspect(self, image, cost=None, route_position=None, side=None):
        """Logs an inspect action and deducts inspect cost."""
        # Set the inspection flag for the appropriate side
        if side == 'left':
            self.inspected_left = True
        elif side == 'right':
            self.inspected_right = True

        if cost is None:
            cost = ActionCost.INSPECT.value
        self.remaining_time -= cost

    #TODO: fix this to be able to scram images and not humanoids + other number tweaks
    def scram(self, image_left, image_right, time_cost=None, route_position=None):
        """
        scrams
        updates scorekeeper
        """
        self.log_consolidated('scram', image_left=image_left, image_right=image_right, route_position=route_position, action_side='both')
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
            # print(f"[SCRAM] Earned ${earnings} for {num_humans} humans.")
        
        self.ambulance["zombie"] = 0
        self.ambulance["injured"] = 0
        self.ambulance["healthy"] = 0
        self.ambulance_people.clear()
        # print(f"[DEBUG] Ambulance cleared: {self.ambulance_people}")
    
    def available_action_space(self):
        """
        returns available action space as a list of bools
        For RL training, this returns the 6-action space: [SKIP_BOTH, SQUISH_LEFT, SQUISH_RIGHT, SAVE_LEFT, SAVE_RIGHT, SCRAM]
        """
        # Check if we're in RL training mode (called from training interface)
        import inspect
        frame = inspect.currentframe()
        is_rl_training = False
        try:
            # Check if we're being called from training_interface
            while frame:
                filename = frame.f_code.co_filename
                if 'training_interface.py' in filename:
                    is_rl_training = True
                    break
                frame = frame.f_back
        except:
            pass
        finally:
            del frame
        
        if is_rl_training:
            # RL 6-action space: [SKIP_BOTH, SQUISH_LEFT, SQUISH_RIGHT, SAVE_LEFT, SAVE_RIGHT, SCRAM]
            actions = [True] * 6  # All actions initially available
            
            # Time constraints
            if self.remaining_time <= 0:
                actions[0] = False  # SKIP_BOTH
                actions[1] = False  # SQUISH_LEFT
                actions[2] = False  # SQUISH_RIGHT
                actions[3] = False  # SAVE_LEFT
                actions[4] = False  # SAVE_RIGHT
                # SCRAM (actions[5]) always available
            
            # Capacity constraints
            if self.at_capacity():
                actions[3] = False  # SAVE_LEFT
                actions[4] = False  # SAVE_RIGHT
                
            return actions
        else:
            # Original 5-action space for UI gameplay
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

    def get_score(self, image_left, image_right):
        self.scram(image_left, image_right)
        return self.scorekeeper
    
    def get_final_score(self, route_complete=True):
        """
        Calculate the final score based on saved/killed and also on time remaining
        """
        score = 0
        # print(f"[Debug] Ambulance: {self.ambulance}, Scorekeeper: {self.scorekeeper}, Remaining Time: {self.remaining_time}")
        score += self.ambulance["healthy"] * SCORE_HEALTHY
        score += self.ambulance["injured"] * SCORE_INJURED
        score += self.ambulance["zombie"] * SCORE_ZOMBIE
        score += self.scorekeeper["killed"] * SCORE_KILLED
        score += self.scorekeeper["saved"] * SCORE_SAVED
        score += self.scorekeeper["zombie_cured"] * 25
        score += self.scorekeeper["human_infected"] * -15
        score += self.scorekeeper["zombie_killed"] * 10
        score += self.scorekeeper["human_killed"] * -10
        score += self.remaining_time * 0.2
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
                    # Keep the original role, don't change it to "blank"

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
                    # Keep the original role, don't change it to "Civilian"
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
            