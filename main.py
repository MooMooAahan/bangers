import argparse
import os
from endpoints.data_parser import DataParser
from endpoints.heuristic_interface import HeuristicInterface
from endpoints.training_interface import TrainInterface
from endpoints.inference_interface import InferInterface
from gameplay.ui import UI, IntroScreen

from gameplay.scorekeeper import ScoreKeeper
from gameplay.enums import ActionCost
from model_training.rl_training import train
import tkinter as tk


class Main(object):
    """
    Base class for the SGAI 2023 game
    """
    def __init__(self, mode, log, args = None):
        self.args = args
        
        # Create timestamped log file if logging is enabled
        if log:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_filename = f"RL-unfiltered-data_{timestamp}.csv"
            print(f"📝 Logging to: {self.log_filename}")
            
            # Create RL-Data directory if it doesn't exist
            os.makedirs('RL-Data', exist_ok=True)
            
            # Initialize empty log file with headers
            headers = "timestamp,local_run_id,route_position,humanoid_class,capacity,remaining_time,role,inspected,action,action_side\n"
            with open(self.log_filename, "w") as f:
                f.write(headers)
        else:
            self.log_filename = None
        
        self.data_fp = os.getenv("SGAI_DATA", default='data')
        self.data_parser = DataParser(self.data_fp)

        shift_length = 720
        capacity = 10
        # Pass the timestamped log filename to ScoreKeeper if logging is enabled
        if self.log_filename:
            self.scorekeeper = ScoreKeeper(shift_length, capacity, log_path=self.log_filename)
        else:
            self.scorekeeper = ScoreKeeper(shift_length, capacity)

        # Create a single Tk root window and hide it
        self.root = tk.Tk()
        self.root.withdraw()

        if mode == 'heuristic':   # Run in background until all humanoids are processed
            simon = HeuristicInterface()
            while len(self.data_parser.unvisited) > 0:
                if self.scorekeeper.remaining_time <= 0:
                    print('Ran out of time')
                    break
                else:
                    # Get both left and right images for junction scenario (like real game)
                    image_left = self.data_parser.get_random(side='left')
                    image_right = self.data_parser.get_random(side='right')
                    
                    # Get AI suggestion for the junction scenario
                    action = simon.get_model_suggestion(image_left, image_right, self.scorekeeper.at_capacity())
                    
                    # Execute the action based on the suggestion
                    if action == ActionCost.SKIP:
                        # Skip both sides of the junction
                        self.scorekeeper.skip_both(image_left, image_right)
                    elif action == ActionCost.SAVE:
                        # For now, always save the left image (can be enhanced later)
                        self.scorekeeper.save(image_left, image_left=image_left, image_right=image_right)
                    elif action == ActionCost.SCRAM:
                        self.scorekeeper.scram(image_left, image_right)
                    else:
                        raise ValueError("Invalid action suggested")

                    # Keep a reference to the last images for final score calculation
                    self.image_left, self.image_right = image_left, image_right
            if log:
                self.scorekeeper.save_log(final=True)
            print("RL equiv reward:",self.scorekeeper.get_cumulative_reward())
            # Create dummy images for score calculation if needed
            if hasattr(self, 'image_left') and hasattr(self, 'image_right'):
                print(self.scorekeeper.get_score(self.image_left, self.image_right))
            else:
                print("Final score calculation skipped - no current images")
        elif mode == 'train':  # RL training script
            train()
        elif mode == 'infer':   # Load trained RL model and run inference
            # Require explicit model path to prevent architecture mismatches
            if not hasattr(self.args, 'model') or self.args.model is None:
                print("❌ Inference mode requires model path:")
                print("Usage: python main.py -m infer --model <model_path>")
                print("Example: python main.py -m infer --model model_training/ZombieRL/PPO_ZombieRL_FINAL_42_49.pth")
                return
            model_path = self.args.model
            simon = InferInterface(self.root, 800, 600, self.data_parser, self.scorekeeper, rl_model_file=model_path, display=False)
            
            # Enforce 20-movement limit like human gameplay
            movement_count = 0
            max_movements = 20
            
            while movement_count < max_movements and len(simon.data_parser.unvisited) > 0:
                if simon.scorekeeper.remaining_time <= 0:
                    break
                else:
                    humanoid = self.data_parser.get_random(side = 'left') ## TODO: this is currently hardcoded to left side
                    simon.act(humanoid)
                    movement_count += 1
                    print(f"Movement {movement_count}/{max_movements}")
                    
            self.scorekeeper = simon.scorekeeper
            if log:
                self.scorekeeper.save_log(final=True)
            print("RL equiv reward:",self.scorekeeper.get_cumulative_reward())
            # Get final score with route_complete=True for proper 20-movement scoring
            final_stats = {
                'movements_completed': f"{movement_count}/{max_movements}",
                'zombies_killed': self.scorekeeper.scorekeeper["zombie_killed"],
                'humans_killed': self.scorekeeper.scorekeeper["human_killed"],
                'zombie_cured': self.scorekeeper.scorekeeper.get("zombie_cured", 0),
                'human_infected': self.scorekeeper.scorekeeper.get("human_infected", 0),
                'people_in_ambulance': sum(self.scorekeeper.ambulance.values()),
                'remaining_time_minutes': self.scorekeeper.remaining_time,
            }
            print("="*50)
            print("🤖 RL INFERENCE RESULTS:")
            print("="*50)
            for key, value in final_stats.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
            
            time_bonus = self.scorekeeper.remaining_time * 0.2
            final_score = self.scorekeeper.get_final_score(route_complete=(movement_count >= max_movements))
            print(f"Time Bonus: {time_bonus} points")
            print(f"FINAL SCORE: {final_score}")
            print("="*50)
            if log:
                # Log the final score to CSV
                self.scorekeeper.log_final_score(
                    final_score=final_score,
                    movement_count=movement_count, 
                    max_movements=max_movements,
                    route_complete=(movement_count >= max_movements)
                )
                # Move completed log file to RL-Data directory
                import shutil
                final_log_path = f"RL-Data/{self.log_filename}"
                shutil.move(self.log_filename, final_log_path)
                print(f"✅ Log saved to {final_log_path}")
                print(f"💡 Run 'python clean_rl_logs.py --clean-all' to process all logs")
        else: # Launch UI gameplay
            def start_ui():
                print("start_ui called")
                try:
                    self.ui = UI(self.data_parser, self.scorekeeper, self.data_fp, False, log, root=self.root)
                    self.root.deiconify()  # Show the main window
                    # Close the intro screen after the main UI is ready
                    if hasattr(self, 'intro_screen'):
                        self.intro_screen.root.destroy()
                except Exception as e:
                    print("Exception in UI creation:", e)

            self.intro_screen = IntroScreen(start_ui, self.root)
            self.intro_screen.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='python3 main.py',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-m', '--mode', type=str, default = 'user', choices = ['user','heuristic','train','infer'],)
    parser.add_argument('-l', '--log', type=bool, default = False)
    parser.add_argument('--model', type=str, help='Path to trained RL model for inference mode')
    # parser.add_argument('-a', '--automode', action='store_true', help='No UI, run autonomously with model suggestions')
    # parser.add_argument('-d', '--disable', action='store_true', help='Disable model help')

    args = parser.parse_args()
    Main(args.mode, args.log, args)