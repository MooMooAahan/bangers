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
    def __init__(self, mode, log):
        # Clear log.csv at game initialization (preserve headers)
        if os.path.exists("log.csv"):
            # Read the first line (headers) and write it back
            with open("log.csv", "r") as f:
                first_line = f.readline().strip()
            with open("log.csv", "w") as f:
                f.write(first_line + "\n")  # Write back only the headers
            print("log.csv cleared at game initialization (headers preserved)")
        
        self.data_fp = os.getenv("SGAI_DATA", default='data')
        self.data_parser = DataParser(self.data_fp)

        shift_length = 720
        capacity = 10
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
        elif mode == 'infer':  # RL training script
            simon = InferInterface(None, None, None, self.data_parser, self.scorekeeper, display=False,)
            while len(simon.data_parser.unvisited) > 0:
                if simon.scorekeeper.remaining_time <= 0:
                    break
                else:
                    humanoid = self.data_parser.get_random(side = 'left') ## TODO: this is currently hardcoded to left side
                    simon.act(humanoid)
            self.scorekeeper = simon.scorekeeper
            if log:
                self.scorekeeper.save_log()
            print("RL equiv reward:",self.scorekeeper.get_cumulative_reward())
            # Skip final score calculation for inference mode (no current images available)
            print("Final score calculation skipped for inference mode")
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
    # parser.add_argument('-a', '--automode', action='store_true', help='No UI, run autonomously with model suggestions')
    # parser.add_argument('-d', '--disable', action='store_true', help='Disable model help')

    args = parser.parse_args()
    Main(args.mode, args.log)