import math
import tkinter as tk
from ui_elements.button_menu import ButtonMenu, LeftButtonMenu, RightButtonMenu
from ui_elements.capacity_meter import CapacityMeter
from ui_elements.clock import Clock
from endpoints.heuristic_interface import HeuristicInterface
from ui_elements.game_viewer import GameViewer
from ui_elements.machine_menu import MachineMenu
from os.path import join
from PIL import Image, ImageTk
import random

class IntroScreen:
    def __init__(self, on_start_callback):
        self.root = tk.Tk()
        self.root.title("Welcome to SGAI 2025 - Team Splice")
        self.root.geometry("600x350")
        self.root.resizable(False, False)
        self.on_start_callback = on_start_callback
        self.setup_ui()
    def setup_ui(self):
        label = tk.Label(self.root, text="Welcome to Beaverworks SGAI 2025 - Team Splice",
                         font=("Helvetica", 18), pady=40)
        label.pack()
        start_button = tk.Button(self.root, text="Start Game", font=("Helvetica", 16), width=15,
                                 command=self.start_game)
        start_button.pack(pady=30)
    def start_game(self):
        self.root.destroy()
        self.on_start_callback()
    def run(self):
        self.root.mainloop()

class UI(object):
    def __init__(self, data_parser, scorekeeper, data_fp, suggest, log):
        # Base window setup
        self.scorekeeper = scorekeeper  # Store scorekeeper reference
        capacity = self.scorekeeper.capacity
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
        self.humanoid_left, self.humanoid_right, scenario_number, scenario_desc = data_parser.get_scenario()
        print(f"[UI DEBUG] Initial Scenario {scenario_number}: left={scenario_desc[0]}, right={scenario_desc[1]}")
        self.log = log
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
                        ("Inspect (15 mins)", lambda: self.show_action_popup("Inspect")),
                        ("Squish (5 mins)", lambda: self.show_action_popup("Squish")),
                        ("Save (30 mins)", lambda: self.show_action_popup("Save")),
                        ("Scram (2 hrs)", lambda: [
                                            print("[DEBUG] Scram penalty applied:", 120 - getattr(self.scorekeeper, "scram_time_reduction", 0), "minutes"),
                                        self.add_elapsed_time(120 - getattr(self.scorekeeper, "scram_time_reduction", 0)),
                                           scorekeeper.scram(self.humanoid_left),
                                           scorekeeper.scram(self.humanoid_right),
                                           self.reset_map(),
                                           self.update_ui(scorekeeper),
                                           self.get_next(
                                               data_fp,
                                               data_parser,
                                               scorekeeper)])]
        self.button_menu = ButtonMenu(self.root, user_buttons)
        # Restore left/right button menus for Squish/Save
        # Add extra button menu with three buttons
        # Save references to left/right button actions for map movement
        self.left_action_callbacks = [
            lambda: [self.add_elapsed_time(15 - getattr(self.scorekeeper, "inspect_cost_reduction", 0)), self.update_ui(scorekeeper), self.check_game_end(data_fp, data_parser, scorekeeper)],  # Inspect Left (no move)
            lambda: [self.add_elapsed_time(5), scorekeeper.squish(self.humanoid_left), self.move_map_left(), self.update_ui(scorekeeper), self.get_next(data_fp, data_parser, scorekeeper)],  # Squish Left
            lambda: [self.add_elapsed_time(30), scorekeeper.save(self.humanoid_left), self.move_map_left(), self.update_ui(scorekeeper), self.get_next(data_fp, data_parser, scorekeeper)]  # Save Left
        ]
        self.left_button_menu = LeftButtonMenu(self.root, [
            ("Inspect Left", self.left_action_callbacks[0]),
            ("Squish Left", self.left_action_callbacks[1]),
            ("Save Left", self.left_action_callbacks[2])
        ])

        self.right_action_callbacks = [
            lambda: [self.add_elapsed_time(15 - getattr(self.scorekeeper, "inspect_cost_reduction", 0)), self.update_ui(scorekeeper), self.check_game_end(data_fp, data_parser, scorekeeper)],  # Inspect Right (no move)
            lambda: [self.add_elapsed_time(5), scorekeeper.squish(self.humanoid_right), self.move_map_right(), self.update_ui(scorekeeper), self.get_next(data_fp, data_parser, scorekeeper)],  # Squish Right
            lambda: [self.add_elapsed_time(30), scorekeeper.save(self.humanoid_right), self.move_map_right(), self.update_ui(scorekeeper), self.get_next(data_fp, data_parser, scorekeeper)]  # Save Right
        ]
        
        self.right_button_menu = RightButtonMenu(self.root, [
            ("Inspect Right", self.right_action_callbacks[0]),
            ("Squish Right", self.right_action_callbacks[1]),
            ("Save Right", self.right_action_callbacks[2])
        ])
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
        init_h = 12
        init_m = 0
        self.clock = Clock(self.root, w, h, init_h, init_m)
        def get_ambulance_people():
            return [f'{k}: {v}' for k, v in self.scorekeeper.ambulance.items() if v > 0]
        self.capacity_meter = CapacityMeter(self.root, w, h, capacity, get_ambulance_contents=get_ambulance_people)
        rules_btn_width = 200
        rules_btn_x = 100
        rules_btn_y = 90
        self.rules_btn = tk.Button(self.root, text="Rules", command=self.show_rules, 
                                  font=("Arial", 18), bg="#4CAF50", fg="white", 
                                  relief="raised", bd=2, width=12)
        self.rules_btn.place(x=rules_btn_x, y=rules_btn_y)

        # 2D grid map setup (bottom left)
        # 1 = up, 2 = down, 3 = right, 4 = left, 5 = base
        self.map_array = [
            [3,3,3,2,3,2],
            [1,1,1,2,1,2],
            [5,2,4,4,1,2],
            [1,3,3,3,1,2],
            [1,4,4,1,1,2],
            [1,1,1,4,4,4],
        ]
        self.grid_rows = len(self.map_array)
        self.grid_cols = len(self.map_array[0])
        self.cell_size = 44  # Small for better fit
        self.grid_origin = (20, 470)  # Lower left, but higher up
        self.create_grid_map_canvas()  # Create the canvas first
        self.reset_map()               # Now it's safe to call reset_map()
        self.draw_grid_map()           # Draw the map with background image
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.direction_idx = 0  # Start facing right
        # self.create_grid_map_canvas() # This line is moved
        # self.draw_grid_map() # This line is moved

        self.upgrade_btn = tk.Button(self.root, text="Upgrade Shop",
                                command=self.show_upgrade_shop,
                                font=("Arial", 18), bg="#F39C12", fg="white",
                                relief="raised", bd=2, width=12)
        self.upgrade_btn.place(x=300, y=rules_btn_y)



        self.root.mainloop()

    def add_elapsed_time(self, minutes):
        """Add elapsed time and update scorekeeper's remaining time"""
        self.elapsed_time += minutes
        remaining_time = self.total_time - self.elapsed_time
        # Update the scorekeeper's remaining time
        self.scorekeeper.remaining_time = remaining_time
    def show_leftright_instructions(self):
        leftright_text = (
            "Make sure to press L or R! You can only do this action for 1 side. \n"
            "If you are inspecting, you can press both L and R to inspect both sides. \n"
        )
        return leftright_text

    def show_action_popup(self, action_name):
        """Show a popup with highlighted instructions when large action buttons are clicked"""
        instructions_text = self.show_leftright_instructions()
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title(f"{action_name} Instructions")
        popup.geometry("700x300")
        popup.resizable(False, False)
        
        # Center the popup on the screen
        popup.transient(self.root)
        popup.grab_set()
        
        # Create main frame
        main_frame = tk.Frame(popup, padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")
        
        # Title label
        if action_name == "Save":
            title_label = tk.Label(main_frame, text=f"Saving", 
                              font=("Arial", 16, "bold"), fg="#2E86AB")
            title_label.pack(pady=(0, 15))
        else:
            title_label = tk.Label(main_frame, text=f"{action_name}ing", 
                              font=("Arial", 16, "bold"), fg="#2E86AB")
            title_label.pack(pady=(0, 15))
        
        # Instructions text with highlighting
        instructions_label = tk.Label(main_frame, text=instructions_text, 
                                    justify="center", anchor="center", font=("Arial", 12),
                                    fg="#E74C3C", bg="#FDF2E9", 
                                    relief="solid", bd=1, padx=15, pady=15)
        instructions_label.pack(expand=True, fill="both", pady=(0, 15))
        
        # Close button
        close_btn = tk.Button(main_frame, text="Close", 
                             command=popup.destroy,
                             font=("Arial", 12), bg="#3498DB", fg="white",
                             relief="raised", bd=2, padx=20, pady=5)
        close_btn.pack()


    def show_rules(self):
        rules_text = (
            "Game Rules:\n"
            "- The goal is to finish the ambulance route by the end of the day\n"
            "- Choose an action: Skip when presented with fiqure ahead.\n"
            "- The goal is to save as many humans and squash  the zombies.\n"
            "- Your choices affect your score \n"
            "- The game ends when your tasks are completed or the day ends. \n"
            #More rules if needed (make sure to add \n)
            "- This is where we can add more rules in case we need to. \n\n"
        )

        rules_window = tk.Toplevel(self.root)
        rules_window.title("Game Rules")
        rules_window.geometry("600x400")
        rules_window.resizable(False, False)

        label = tk.Label(rules_window, text=rules_text, justify="left", padx=10, pady=10, font=("Helvetica", 11))
        label.pack(expand=True, fill="both")

        close_btn = tk.Button(rules_window, text="Close", command=rules_window.destroy)
        close_btn.pack(pady=10)

    def update_ui(self, scorekeeper):     
        # Convert elapsed_time to clock positions
        # elapsed_time mod 60 gives us the minute position
        # elapsed_time / 60 gives us the hour position
        self.elapsed_time = self.total_time - scorekeeper.remaining_time
        hours_elapsed = math.floor(self.elapsed_time / 60.0)
        minutes_elapsed = self.elapsed_time % 60
        
        # Convert to 12-hour clock format (1-12)
        # 0 hours = 12 o'clock, 1 hour = 1 o'clock, etc.
        h = (hours_elapsed % 12)
        if h == 0:
            h = 12
        m = minutes_elapsed
        
        print(f"I changed my time :> to {h} {m}")
        
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
            self.hide_map = True # Hide the map when game ends
            self.draw_grid_map() # Redraw the map to hide it
        else:
            self.humanoid_left, self.humanoid_right, scenario_number, scenario_desc = data_parser.get_scenario()
            print(f"[UI DEBUG] Scenario {scenario_number}: left={scenario_desc[0]}, right={scenario_desc[1]}")
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
        
        # Also disable the left and right button menus with the same logic
        self.left_button_menu.disable_buttons(remaining_time, remaining_humanoids, at_capacity)
        self.right_button_menu.disable_buttons(remaining_time, remaining_humanoids, at_capacity)
        
            
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
        data_parser.reset()
        self.humanoid_left, self.humanoid_right, scenario_number, scenario_desc = data_parser.get_scenario()
        print(f"[UI DEBUG] Reset Scenario {scenario_number}: left={scenario_desc[0]}, right={scenario_desc[1]}")
        fp_left = join(data_fp, self.humanoid_left.fp)
        fp_right = join(data_fp, self.humanoid_right.fp)
        self.game_viewer_left.create_photo(fp_left)
        self.game_viewer_right.create_photo(fp_right)
        self.update_ui(self.scorekeeper)
        self.button_menu.enable_all_buttons()
        self.left_button_menu.enable_all_buttons()
        self.right_button_menu.enable_all_buttons()
        self.clock.update_time(12, 0)
        self.reset_map() # Reset map on game end
        self.hide_map = False
        self.draw_grid_map()
        self.map_canvas.place(x=self.grid_origin[0], y=self.grid_origin[1]) # Always re-place the map after replay
        # Clear any widgets in both canvases
        for widget in self.game_viewer_left.canvas.pack_slaves():
            widget.destroy()
        for widget in self.game_viewer_right.canvas.pack_slaves():
            widget.destroy()
        # End of canvas widget clearing
    def create_grid_map_canvas(self):
        w = self.grid_cols * self.cell_size + 2 * 10
        h = self.grid_rows * self.cell_size + 2 * 10
        self.map_canvas = tk.Canvas(self.root, width=w, height=h, bg="white", highlightthickness=1, highlightbackground="#888")
        self.map_canvas.place(x=self.grid_origin[0], y=self.grid_origin[1])
    def draw_grid_map(self):
        if hasattr(self, 'map_canvas') and getattr(self, 'hide_map', False):
            self.map_canvas.place_forget()
            return
        self.map_canvas.delete("all")
        
        
        # Image of map
        # Use calculated canvas dimensions since winfo_width/height return 0 during initial creation
        canvas_width = self.grid_cols * self.cell_size + 20
        canvas_height = self.grid_rows * self.cell_size + 20
        bg_img = ImageTk.PhotoImage(Image.open('./ChatgptMap.png').resize((canvas_width, canvas_height)))
        self.map_canvas.create_image(0, 0, anchor=tk.NW, image=bg_img)
        self.bg_img = bg_img  # Keep a reference to avoid garbage collection
       
        
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                val = self.map_array[r][c]
                x1 = c * self.cell_size + 10
                y1 = r * self.cell_size + 10
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
        # Draw path
        if len(self.path_history) > 1:
            points = []
            for (pr, pc) in self.path_history:
                px = pc * self.cell_size + 10 + self.cell_size // 2
                py = pr * self.cell_size + 10 + self.cell_size // 2
                points.extend([px, py])
            if len(points) >= 4:
                self.map_canvas.create_line(*points, fill="#3498db", width=5, smooth=True)
        elif len(self.path_history) == 1:
            pr, pc = self.ambulance_pos
            px = pc * self.cell_size + 10 + self.cell_size // 2
            py = pr * self.cell_size + 10 + self.cell_size // 2
            self.map_canvas.create_oval(px-4, py-4, px+4, py+4, fill="#3498db", outline="")
        # Draw ambulance at current position
        r, c = self.ambulance_pos
        x = c * self.cell_size + 10 + self.cell_size // 2
        y = r * self.cell_size + 10 + self.cell_size // 2
        self.map_canvas.create_oval(x-18, y-18, x+18, y+18, fill="#fff", outline="#3498db", width=3)
        self.map_canvas.create_text(x, y, text="🚑", font=("Arial", 18))

    def move_ambulance_by_cell(self):
        r, c = self.ambulance_pos
        val = self.map_array[r][c]
        if val == 1:  # up
            new_r, new_c = r-1, c
        elif val == 2:  # down
            new_r, new_c = r+1, c
        elif val == 3:  # right
            new_r, new_c = r, c+1
        elif val == 4:  # left
            new_r, new_c = r, c-1
        elif val == 5:  # base (move up)
            new_r, new_c = r-1, c
        else:
            new_r, new_c = r, c
        if 0 <= new_r < self.grid_rows and 0 <= new_c < self.grid_cols and self.map_array[new_r][new_c] != 0:
            self.ambulance_pos = [new_r, new_c]
            self.path_history.append(tuple(self.ambulance_pos))
        self.draw_grid_map()

    def move_map_left(self):
        self.move_ambulance_by_cell()

    def move_map_right(self):
        self.move_ambulance_by_cell()

    def reset_map(self):
        self.hide_map = False  # Ensure map is visible after replay/reset
        # Always start on BASE cell (value 5), then 3, then first walkable
        base_r, base_c = None, None
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if self.map_array[r][c] == 5:
                    base_r, base_c = r, c
                    break
            if base_r is not None:
                break
        if base_r is None or base_c is None:
            # Try to find a cell with value 3 (right)
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    if self.map_array[r][c] == 3:
                        base_r, base_c = r, c
                        break
                if base_r is not None:
                    break
        if base_r is None or base_c is None:
            # Fallback: first walkable cell
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    if self.map_array[r][c] in (1,2,3,4,5):
                        base_r, base_c = r, c
                        break
                if base_r is not None:
                    break
        if base_r is None or base_c is None:
            base_r, base_c = 0, 0
        self.ambulance_pos = [base_r, base_c]
        self.path_history = [tuple(self.ambulance_pos)]
        self.draw_grid_map()

    def show_upgrade_shop(self):
        shop = tk.Toplevel(self.root)
        shop.title("Upgrade Shop")
        shop.geometry("400x300")
        shop.resizable(False, False)

    # Show current money
        money = self.scorekeeper.upgrade_manager.get_money()
        tk.Label(shop, text=f"Money: ${money}", font=("Arial", 16)).pack(pady=10)

    # Show each upgrade
        for name, info in self.scorekeeper.upgrade_manager.upgrades.items():
            upgrade_label = name.replace("_", " ").title()
            level = info["level"]
            cost = info["cost"]

            def make_purchase(n=name):
                if self.scorekeeper.upgrade_manager.purchase(n):
                    shop.destroy()
                    self.show_upgrade_shop()  # Refresh the popup

            btn_text = f"{upgrade_label} (Level {level}) - ${cost}"
            tk.Button(shop, text=btn_text, font=("Arial", 12),
                  command=make_purchase).pack(pady=5)


