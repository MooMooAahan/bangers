import math
import tkinter as tk
from ui_elements.button_menu import ButtonMenu
from ui_elements.capacity_meter import CapacityMeter
from ui_elements.clock import Clock
from endpoints.heuristic_interface import HeuristicInterface
from ui_elements.game_viewer import GameViewer
from ui_elements.machine_menu import MachineMenu
from os.path import join


import math
import tkinter as tk
from ui_elements.button_menu import ButtonMenu
from ui_elements.capacity_meter import CapacityMeter
from ui_elements.clock import Clock
from endpoints.heuristic_interface import HeuristicInterface
from ui_elements.game_viewer import GameViewer
from ui_elements.machine_menu import MachineMenu
from os.path import join


class UI(object):
    def __init__(self, data_parser, scorekeeper, data_fp, suggest, log):
        # Base window setup
        capacity = 10
        w, h = 1280, 800
        self.root = tk.Tk()
        self.root.title("Beaverworks SGAI 2025 - Team Splice")
        self.root.geometry(f"{w}x{h}")
        self.root.resizable(False, False)

        # Time management variables
        self.total_time = 720  # 12 hours in minutes
        self.elapsed_time = 0  # Start at 0 elapsed time
        self.scorekeeper = scorekeeper  # Store scorekeeper reference
        
        self.humanoid = data_parser.get_random()
        self.log = log

        if suggest:
            self.machine_interface = HeuristicInterface(self.root, w, h)

        #  Add buttons and logo
        user_buttons = [("Skip (15 mins)", lambda: [self.add_elapsed_time(15),
                                          scorekeeper.skip(self.humanoid),
                                          self.update_ui(scorekeeper),
                                          self.get_next(
                                              data_fp,
                                              data_parser,
                                              scorekeeper)]),
                        ("Inspect (15 mins)", lambda: [self.add_elapsed_time(15),
                                             self.update_ui(scorekeeper),
                                             self.check_game_end(data_fp, data_parser, scorekeeper)]),
                        ("Squish (5 mins)", lambda: [self.add_elapsed_time(5),
                                            scorekeeper.squish(self.humanoid),
                                            self.update_ui(scorekeeper),
                                            self.get_next(
                                                data_fp,
                                                data_parser,
                                                scorekeeper)]),
                        ("Save (30 mins)", lambda: [self.add_elapsed_time(30),
                                          scorekeeper.save(self.humanoid),
                                          self.update_ui(scorekeeper),
                                          self.get_next(
                                              data_fp,
                                              data_parser,
                                              scorekeeper)]),
                        ("Scram (2 hrs)", lambda: [self.add_elapsed_time(120),
                                           scorekeeper.scram(self.humanoid),
                                           self.update_ui(scorekeeper),
                                           self.get_next(
                                               data_fp,
                                               data_parser,
                                               scorekeeper)])]
        self.button_menu = ButtonMenu(self.root, user_buttons)

        if suggest:
            machine_buttons = [
                ("Suggest", lambda: [self.machine_interface.suggest(self.humanoid)]),
                ("Act", lambda: [self.machine_interface.act(scorekeeper, self.humanoid),
                                 self.update_ui(scorekeeper),
                                 self.get_next(data_fp, data_parser, scorekeeper)])
            ]
            self.machine_menu = MachineMenu(self.root, machine_buttons)

        # Display central photo
        self.game_viewer = GameViewer(self.root, w, h, data_fp, self.humanoid)
        self.root.bind("<Delete>", self.game_viewer.delete_photo)

        # Display the countdown
        # At start, no time has elapsed, so clock should be at 12 o'clock
        init_h = 12
        init_m = 0
        
        self.clock = Clock(self.root, w, h, init_h, init_m)

        # Display ambulance capacity
        self.capacity_meter = CapacityMeter(self.root, w, h, capacity)

        self.root.mainloop()

    def add_elapsed_time(self, minutes):
        """Add elapsed time and update scorekeeper's remaining time"""
        self.elapsed_time += minutes
        remaining_time = self.total_time - self.elapsed_time
        # Update the scorekeeper's remaining time
        self.scorekeeper.remaining_time = remaining_time

    def show_rules(self):
        rules_text = (
            "Game Rules:\n"
            "- The goal is to finish the ambulance route by the end of the day\n"
            "- Choose an action: Skip when presented with fiqure ahead.\n"
            "- The goal is to save as many humans and squash  the zombies.\n"
            "- Your choices affect your score \n"
            "- The game ends when your tasks are completed or the day ends"
        )

        rules_window = tk.Toplevel(self.root)
        rules_window.title("Game Rules")
        rules_window.geometry("400x300")
        rules_window.resizable(False, False)

        label = tk.Label(rules_window, text=rules_text, justify="left", padx=10, pady=10, font=("Helvetica", 11))
        label.pack(expand=True, fill="both")

        close_btn = tk.Button(rules_window, text="Close", command=rules_window.destroy)
        close_btn.pack(pady=10)

    def update_ui(self, scorekeeper):     
        # Convert elapsed_time to clock positions
        # elapsed_time mod 60 gives us the minute position
        # elapsed_time / 60 gives us the hour position
        hours_elapsed = math.floor(self.elapsed_time / 60.0)
        minutes_elapsed = self.elapsed_time % 60
        
        # Convert to 12-hour clock format (1-12)
        # 0 hours = 12 o'clock, 1 hour = 1 o'clock, etc.
        h = (hours_elapsed % 12)
        if h == 0:
            h = 12
        m = minutes_elapsed
        
        self.clock.update_time(h, m)
        self.capacity_meter.update_fill(scorekeeper.get_current_capacity())

    def on_resize(self, event):
        w = 0.6 * self.root.winfo_width()
        h = 0.7 * self.root.winfo_height()
        self.game_viewer.canvas.config(width=w, height=h)

    def get_next(self, data_fp, data_parser, scorekeeper):
        remaining = len(data_parser.unvisited)
        remaining_time = self.total_time - self.elapsed_time

        # Ran out of humanoids or time? End game
        if remaining == 0 or remaining_time <= 0:
            if self.log:
                scorekeeper.save_log()
            self.capacity_meter.update_fill(0)
            self.game_viewer.delete_photo(None)
            final_score = scorekeeper.get_final_score()
            accuracy = round(scorekeeper.get_accuracy() * 100, 2)
            self.game_viewer.display_score(scorekeeper.get_score(), final_score, accuracy)

        else:
            self.humanoid = data_parser.get_random()
            fp = join(data_fp, self.humanoid.fp)
            self.game_viewer.create_photo(fp)

        # Disable buttons that would exceed time limit
        self.disable_buttons_if_insufficient_time(remaining_time, remaining, scorekeeper.at_capacity())

    def check_game_end(self, data_fp, data_parser, scorekeeper):
        """Check if game should end due to time running out"""
        remaining_time = self.total_time - self.elapsed_time
        if remaining_time <= 0:
            if self.log:
                scorekeeper.save_log()
            self.capacity_meter.update_fill(0)
            self.game_viewer.delete_photo(None)
            self.game_viewer.display_score(scorekeeper.get_score())
            # Disable all buttons when game ends
            self.disable_buttons_if_insufficient_time(0, 0, False)

    def disable_buttons_if_insufficient_time(self, remaining_time, remaining_humanoids, at_capacity):
        """Disable buttons based on remaining time and other constraints"""
        # Use the existing ButtonMenu.disable_buttons method
        # Note: ButtonMenu now handles Skip (index 0), Inspect (index 1), Squish (index 2), Save (index 3)
        self.button_menu.disable_buttons(remaining_time, remaining_humanoids, at_capacity)
