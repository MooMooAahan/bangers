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
        self.root.title("Beaverworks SGAI 2023 - Dead or Alive")
        self.root.geometry(f"{w}x{h}")
        self.root.resizable(False, False)

        self.humanoid = data_parser.get_random()
        self.log = log

        if suggest:
            self.machine_interface = HeuristicInterface(self.root, w, h)

        # Add "Rules" button at the top center
        rules_button = tk.Button(self.root, text="Rules", command=self.show_rules)
        button_width = 80
        x_pos = (w - button_width) // 2
        rules_button.place(x=x_pos, y=10, width=button_width, height=30)

        # Add user action buttons
        user_buttons = [
            ("Skip", lambda: [scorekeeper.skip(self.humanoid),
                              self.update_ui(scorekeeper),
                              self.get_next(data_fp, data_parser, scorekeeper)]),
            ("Squish", lambda: [scorekeeper.squish(self.humanoid),
                                self.update_ui(scorekeeper),
                                self.get_next(data_fp, data_parser, scorekeeper)]),
            ("Save", lambda: [scorekeeper.save(self.humanoid),
                              self.update_ui(scorekeeper),
                              self.get_next(data_fp, data_parser, scorekeeper)]),
            ("Scram", lambda: [scorekeeper.scram(self.humanoid),
                               self.update_ui(scorekeeper),
                               self.get_next(data_fp, data_parser, scorekeeper)])
        ]
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

        # Display the countdown clock
        init_h = (12 - math.floor(scorekeeper.remaining_time / 60.0))
        init_m = 60 - (scorekeeper.remaining_time % 60)
        self.clock = Clock(self.root, w, h, init_h, init_m)

        # Display ambulance capacity
        self.capacity_meter = CapacityMeter(self.root, w, h, capacity)

        self.root.mainloop()

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
        h = (12 - math.floor(scorekeeper.remaining_time / 60.0))
        m = 60 - (scorekeeper.remaining_time % 60)
        self.clock.update_time(h, m)

        self.capacity_meter.update_fill(scorekeeper.get_current_capacity())

    def on_resize(self, event):
        w = 0.6 * self.root.winfo_width()
        h = 0.7 * self.root.winfo_height()
        self.game_viewer.canvas.config(width=w, height=h)

    def get_next(self, data_fp, data_parser, scorekeeper):
        remaining = len(data_parser.unvisited)

        if remaining == 0 or scorekeeper.remaining_time <= 0:
            if self.log:
                scorekeeper.save_log()
            self.capacity_meter.update_fill(0)
            self.game_viewer.delete_photo(None)
            self.game_viewer.display_score(scorekeeper.get_score())
        else:
            self.humanoid = data_parser.get_random()
            fp = join(data_fp, self.humanoid.fp)
            self.game_viewer.create_photo(fp)

        self.button_menu.disable_buttons(scorekeeper.remaining_time, remaining, scorekeeper.at_capacity())


