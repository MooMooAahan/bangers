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
        
        # Track two humanoids for two images
        self.humanoid_left = data_parser.get_random()
        self.humanoid_right = data_parser.get_random()
        self.log = log
        
        #replay button
        self.replay_btn = None
        
        #replay button
        self.replay_btn = None

        if suggest:
            self.machine_interface = HeuristicInterface(self.root, w, h)

        #  Add buttons and logo
        user_buttons = [("Skip (15 mins)", lambda: [self.add_elapsed_time(15),
                                              scorekeeper.skip(self.humanoid_left),
                                              scorekeeper.skip(self.humanoid_right),
                                              self.update_ui(scorekeeper),
                                              self.get_next(
                                                  data_fp,
                                                  data_parser,
                                                  scorekeeper)]),
                        ("Inspect (15 mins)", lambda: [self.add_elapsed_time(15),
                                                 self.update_ui(scorekeeper),
                                                 self.check_game_end(data_fp, data_parser, scorekeeper)]),
                        ("Squish (5 mins)", lambda: [self.add_elapsed_time(5),
                                            scorekeeper.squish(self.humanoid_left),
                                            scorekeeper.squish(self.humanoid_right),
                                            self.update_ui(scorekeeper),
                                            self.get_next(
                                                data_fp,
                                                data_parser,
                                                scorekeeper)]),
                        ("Save (30 mins)", lambda: [self.add_elapsed_time(30),
                                          scorekeeper.save(self.humanoid_left),
                                          scorekeeper.save(self.humanoid_right),
                                          self.update_ui(scorekeeper),
                                          self.get_next(
                                              data_fp,
                                              data_parser,
                                              scorekeeper)]),
                        ("Scram (2 hrs)", lambda: [self.add_elapsed_time(120),
                                           scorekeeper.scram(self.humanoid_left),
                                           scorekeeper.scram(self.humanoid_right),
                                           self.update_ui(scorekeeper),
                                           self.get_next(
                                               data_fp,
                                               data_parser,
                                               scorekeeper)])]
        self.button_menu = ButtonMenu(self.root, user_buttons)
        
        if suggest:
            machine_buttons = [
                ("Suggest", lambda: [self.machine_interface.suggest(self.humanoid_left)]),
                ("Act", lambda: [self.machine_interface.act(scorekeeper, self.humanoid_left),
                                 self.update_ui(scorekeeper),
                                 self.get_next(data_fp, data_parser, scorekeeper)])
            ]
            self.machine_menu = MachineMenu(self.root, machine_buttons)

        # Display two stacked (vertically) photos, centered horizontally
        image_width = 300
        vertical_gap = 30  # pixels between images
        w, h = 1280, 800
        # Calculate total width for both images side by side
        total_width = image_width * 2
        # Center the pair of images
        center_x = (w - total_width) // 2
        y_top = 100 + 50  # Shift down by 50 pixels
        # Place left and right images side by side
        self.game_viewer_left = GameViewer(self.root, image_width, h, data_fp, self.humanoid_left)
        self.game_viewer_right = GameViewer(self.root, image_width, h, data_fp, self.humanoid_right)
        # Place the canvases - left on the left, right on the right, both at y_top
        self.game_viewer_left.canvas.place(x=center_x, y=y_top)
        self.game_viewer_right.canvas.place(x=center_x + image_width, y=y_top)
        self.root.bind("<Delete>", self.game_viewer_left.delete_photo)
        self.root.bind("<Delete>", self.game_viewer_right.delete_photo)

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
        # Update both game viewers for dual screen setup
        self.game_viewer_left.canvas.config(width=w, height=h)
        self.game_viewer_right.canvas.config(width=w, height=h)

    def get_next(self, data_fp, data_parser, scorekeeper):
        remaining = len(data_parser.unvisited)
        remaining_time = self.total_time - self.elapsed_time

        # Ran out of humanoids or time? End game
        if remaining == 0 or remaining_time <= 0:
            if self.log:
                scorekeeper.save_log()
            self.capacity_meter.update_fill(0)
            self.game_viewer_left.delete_photo(None)
            self.game_viewer_right.delete_photo(None)
            final_score = scorekeeper.get_final_score()
            accuracy = round(scorekeeper.get_accuracy() * 100, 2)
            # Remove any previous final score frame
            if hasattr(self, 'final_score_frame') and self.final_score_frame:
                self.final_score_frame.destroy()
            # Create a new frame for the final score block
            self.final_score_frame = tk.Frame(self.root, width=300, height=300)
            self.final_score_frame.place(relx=0.5, rely=0.5, y=-100, anchor=tk.CENTER)  # Center in the whole window, shifted up 100px
            # 'Game Complete' label
            game_complete_label = tk.Label(self.final_score_frame, text="Game Complete", font=("Arial", 40))
            game_complete_label.pack(pady=(10, 5))
            # 'Final Score' label
            final_score_label = tk.Label(self.final_score_frame, text="FINAL SCORE", font=("Arial", 16))
            final_score_label.pack(pady=(5, 2))
            # Scoring details
            killed_label = tk.Label(self.final_score_frame, text=f"Killed {scorekeeper.get_score()['killed']}", font=("Arial", 12))
            killed_label.pack()
            saved_label = tk.Label(self.final_score_frame, text=f"Saved {scorekeeper.get_score()['saved']}", font=("Arial", 12))
            saved_label.pack()
            score_label = tk.Label(self.final_score_frame, text=f"Final Score: {final_score}", font=("Arial", 12))
            score_label.pack()
            accuracy_label = tk.Label(self.final_score_frame, text=f"Accuracy: {accuracy:.2f}%", font=("Arial", 12))
            accuracy_label.pack()
            # Replay button
            self.replay_btn = tk.Button(self.final_score_frame, text="Replay", command=lambda: self.reset_game(data_parser, data_fp))
            self.replay_btn.pack(pady=(10, 0))
            # Remove any content from the right canvas
            self.game_viewer_right.canvas.delete('all')
        else:
            self.humanoid_left = data_parser.get_random()
            self.humanoid_right = data_parser.get_random()
            fp_left = join(data_fp, self.humanoid_left.fp)
            fp_right = join(data_fp, self.humanoid_right.fp)
            self.game_viewer_left.create_photo(fp_left)
            self.game_viewer_right.create_photo(fp_right)

        # Disable buttons that would exceed time limit
        self.disable_buttons_if_insufficient_time(remaining_time, remaining, scorekeeper.at_capacity())

    def check_game_end(self, data_fp, data_parser, scorekeeper):
        """Check if game should end due to time running out"""
        remaining_time = self.total_time - self.elapsed_time
        if remaining_time <= 0:
            if self.log:
                scorekeeper.save_log()
            self.capacity_meter.update_fill(0)
            self.game_viewer_left.delete_photo(None)
            self.game_viewer_right.delete_photo(None)
            
            final_score = scorekeeper.get_final_score()
            accuracy = round(scorekeeper.get_accuracy() * 100, 2)
            self.game_viewer_left.display_score(scorekeeper.get_score(), final_score, accuracy)
            # Clear the right box and show a message
            self.game_viewer_right.canvas.delete('all')
            # Get canvas dimensions for positioning
            canvas_width = self.game_viewer_right.canvas.winfo_width()
            canvas_height = self.game_viewer_right.canvas.winfo_height()
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            
            # Create "Game Complete" text
            self.game_viewer_right.canvas.create_text(
                center_x,
                center_y,
                text="Game Complete",
                font=("Arial", 20),
                fill="black"
            )
            
            # Place replay button on top of the "Game Complete" text
            # Get the canvas position on the window
            canvas_x = self.game_viewer_right.canvas.winfo_x()
            canvas_y = self.game_viewer_right.canvas.winfo_y()
            # Position button at the center of the right canvas
            button_x = canvas_x + center_x - 50  # Center the button (assuming button width ~100px)
            button_y = canvas_y + center_y + 30  # Position below the text
            self.replay_btn = tk.Button(self.root, text="Replay", command=lambda: self.reset_game(data_parser, data_fp))
            self.replay_btn.place(x=button_x, y=button_y)
            # Disable all buttons when game ends
            self.disable_buttons_if_insufficient_time(0, 0, False)

    def disable_buttons_if_insufficient_time(self, remaining_time, remaining_humanoids, at_capacity):
        """Disable buttons based on remaining time and other constraints"""
        # Use the existing ButtonMenu.disable_buttons method
        # Note: ButtonMenu now handles Skip (index 0), Inspect (index 1), Squish (index 2), Save (index 3)
        self.button_menu.disable_buttons(remaining_time, remaining_humanoids, at_capacity)
        
            
    def reset_game(self, data_parser, data_fp):
        """Restart games"""
        # Remove the final score frame if it exists
        if hasattr(self, 'final_score_frame') and self.final_score_frame:
            self.final_score_frame.destroy()
            self.final_score_frame = None
        # Remove the replay button if it exists and is not in the frame
        if self.replay_btn:
            try:
                self.replay_btn.place_forget()
            except Exception:
                pass
            self.replay_btn = None
        self.elapsed_time = 0
        self.scorekeeper.reset()
        self.humanoid_left = data_parser.get_random()
        self.humanoid_right = data_parser.get_random()
        fp_left = join(data_fp, self.humanoid_left.fp)
        fp_right = join(data_fp, self.humanoid_right.fp)
        self.game_viewer_left.create_photo(fp_left)
        self.game_viewer_right.create_photo(fp_right)
        self.update_ui(self.scorekeeper)
        self.button_menu.enable_all_buttons()
        self.clock.update_time(12, 0)
        # Clear any widgets in both canvases
        for widget in self.game_viewer_left.canvas.pack_slaves():
            widget.destroy()
        for widget in self.game_viewer_right.canvas.pack_slaves():
            widget.destroy()
        data_parser.reset()



