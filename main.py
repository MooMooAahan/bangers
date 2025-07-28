import argparse
import os
from endpoints.data_parser import DataParser
from endpoints.heuristic_interface import HeuristicInterface
from endpoints.training_interface import TrainInterface
from endpoints.inference_interface import InferInterface
from gameplay.ui import UI, IntroScreen

from gameplay.scorekeeper import ScoreKeeper
from gameplay.ui import UI
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
            simon = HeuristicInterface(None, None, None, display = False)
            while len(self.data_parser.unvisited) > 0:
                if self.scorekeeper.remaining_time <= 0:
                    print('Ran out of time')
                    break
                else:
                    image = self.data_parser.get_random(side = 'left') ## TODO: this is currently hardcoded to left side
                    # Fix: Use image.Filename instead of image.fp for Image objects
                    action = simon.get_model_suggestion(image, self.scorekeeper.at_capacity())
                    if action == ActionCost.SKIP:
                        self.scorekeeper.skip(image)
                    elif action == ActionCost.SQUISH:
                        self.scorekeeper.squish(image)
                    elif action == ActionCost.SAVE:
                        self.scorekeeper.save(image)
                    elif action == ActionCost.SCRAM:
                        # Pass image for both left and right (or duplicate if only one side in heuristic mode)
                        self.scorekeeper.scram(image, image)
                    else:
                        raise ValueError("Invalid action suggested")
            if log:
                self.scorekeeper.save_log()
            print("RL equiv reward:",self.scorekeeper.get_cumulative_reward())
            print(self.scorekeeper.get_score())
        elif mode == 'train':  # RL training script
            env = TrainInterface(None, None, None, self.data_parser, self.scorekeeper, display=False,)
            train(env)
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
            print(self.scorekeeper.get_score())
        else: # Launch UI gameplay
            def start_ui():
                print("start_ui called")
                try:
                    self.root.deiconify()
                    self.ui = UI(self.data_parser, self.scorekeeper, self.data_fp, False, log, root=self.root)
                except Exception as e:
                    print("Exception in UI creation:", e)

            intro = IntroScreen(start_ui, self.root)
            intro.run()

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