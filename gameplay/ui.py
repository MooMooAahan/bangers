import math
import tkinter as tk
from ui_elements.button_menu import ButtonMenu, LeftButtonMenu, RightButtonMenu
from ui_elements.capacity_meter import CapacityMeter
from ui_elements.clock import Clock
from endpoints.heuristic_interface import HeuristicInterface
from endpoints.enhanced_predictor import EnhancedPredictor, ImperfectCNNDisplay
from ui_elements.game_viewer import GameViewer
from ui_elements.machine_menu import MachineMenu
from gameplay.scorekeeper import _safe_show_popup
from os.path import join
from PIL import Image, ImageTk
import random
from ui_elements.theme import COLOR_SAVE, COLOR_SKIP, COLOR_SCRAM, COLOR_SQUISH, COLOR_INSPECT
import os
from data_uploader import DataUploader



class IntroScreen:
    def __init__(self, on_start_callback, root):
        self.root = tk.Toplevel(root)
        self.root.title("Welcome to SGAI 2025 - Team Splice")
        window_width = 600
        window_height = 350
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
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
    def __init__(self, data_parser, scorekeeper, data_fp, suggest, log, root):

        # Initialize data uploader
        self.data_uploader = DataUploader()
        
        # Base window setup
        self.data_parser = data_parser  # Store data_parser reference
        self.data_fp = data_fp  # Store data_fp reference
        self.scorekeeper = scorekeeper  # Store scorekeeper reference
        capacity = self.scorekeeper.capacity
        self.false_saves = 0
        w, h = 1280, 800
        self.root = root  # Use the passed root instead of creating a new one
        self.root.configure(bg="black")
        self.create_menu_bar()
        self.root.title("Beaverworks SGAI 2025 - Team Splice")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (w // 2)
        y = (screen_height // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.resizable(False, False)
        self.false_saves = 0 
        # Time management variables
        self.total_time = 720  # 12 hours in minutes
        # self.elapsed_time removed; time is managed by ScoreKeeper
        self.scorekeeper = scorekeeper  # Store scorekeeper reference
        
        # Movement tracking
        self.movement_count = 0  # Track number of ambulance movements
        self.route_complete = False  # Flag to track if route is complete
  
        #TODO: fix to be getting images, not humanoids
        self.image_left, self.image_right = data_parser.get_random(side='left'), data_parser.get_random(side='right')
        # Track two humanoids for two images
        # self.humanoid_left, self.humanoid_right, scenario_number, scenario_desc, self.scenario_humanoid_attributes = data_parser.get_scenario()
        # print(f"[UI DEBUG] Initial Scenario {scenario_number}: left={scenario_desc[0]}, right={scenario_desc[1]}")
        self.log = log
        #replay button
        self.replay_btn = None
        
        # Initialize enhanced predictor and imperfect CNN display
        self.enhanced_predictor = EnhancedPredictor()
        self.imperfect_cnn = ImperfectCNNDisplay(self.enhanced_predictor, accuracy_rate=0.5)
        
        if suggest:
            self.machine_interface = HeuristicInterface(self.root, w, h)
        #  Add buttons and logo
        def get_scram_time():
            base_time = max(5, self.movement_count * 5)
            reduction = getattr(self.scorekeeper, "scram_time_reduction", 0)
            return max(5, base_time - reduction)
        def get_scram_text():
            return f"Scram ({get_scram_time()} mins)"
        # Remove get_inspect_time and get_inspect_text functions entirely.
        # In the main button menu:
        def get_inspect_cost():
            base_cost = 15
            reduction = getattr(self.scorekeeper, "inspect_cost_reduction", 0)
            return max(5, base_cost - reduction)
        # In left/right button menus:
        def get_inspect_cost_left_right():
            base_cost = 15
            reduction = getattr(self.scorekeeper, "inspect_cost_reduction", 0)
            return max(5, base_cost - reduction)

        # We'll need to update the button text dynamically, so store the button objects
        self.user_buttons = [
            ("Skip (15 mins)", lambda: [
                                  scorekeeper.skip_both(self.image_left, self.image_right, route_position=self.movement_count),
                                  self.move_ambulance_by_cell(),
                                  self.update_ui(scorekeeper),
                                  self.get_next(
                                      data_fp,
                                      data_parser,
                                      scorekeeper) if not getattr(self, 'route_complete', False) else None],COLOR_SKIP),
            (f"Inspect ({get_inspect_cost()} mins)", lambda: [self.show_action_popup("Inspect")],COLOR_INSPECT),
            ("Squish (5 mins)", lambda: self.show_action_popup("Squish"),COLOR_SQUISH),
            ("Save (30 mins)", lambda: self.show_action_popup("Save"),COLOR_SAVE),
            (get_scram_text(), lambda: [
                                    # print(f"[DEBUG] Scram penalty applied: {get_scram_time()} minutes"),
                                    scorekeeper.scram(self.image_left, self.image_right, time_cost=get_scram_time(), route_position=self.movement_count),
                                    self.move_ambulance_by_cell(),
                                    self.update_ui(scorekeeper),
                                    self.get_next(
                                        data_fp,
                                        data_parser,
                                        scorekeeper) if not getattr(self, 'route_complete', False) else None],COLOR_SCRAM)
        ]

        # Debug and try/except for each major widget
        try:
            print("Creating button menu")
            self.button_menu = ButtonMenu(self.root, self.user_buttons)
            print("Button menu created")
        except Exception as e:
            print("Exception creating button menu:", e)

        # Patch: update Scram and Inspect button text after every action

        def update_button_texts():
            # Button order: Skip, Inspect, Squish, Save, Scram
            self.button_menu.buttons[1].config(text=f"Inspect ({get_inspect_cost()} mins)")
            self.button_menu.buttons[4].config(text=get_scram_text())
        self.update_button_texts = update_button_texts

        # Call update_button_texts after every UI update
        orig_update_ui = self.update_ui
        def patched_update_ui(scorekeeper):
            orig_update_ui(scorekeeper)
            self.update_button_texts()
        self.update_ui = patched_update_ui

        # Restore left/right button menus for Squish/Save
        # Add extra button menu with three buttons
        # Save references to left/right button actions for map movement
        self.left_action_callbacks = [
            lambda: [self.print_scenario_side_attributes('left'), self.scorekeeper.inspect(self.image_left, cost=get_inspect_cost_left_right(), route_position=self.movement_count, side='left'), self.update_ui(self.scorekeeper), self.check_game_end(data_fp, data_parser, self.scorekeeper)],  # Inspect Left
            lambda: [self.scorekeeper.squish(self.image_left, route_position=self.movement_count, side='left', image_left=self.image_left, image_right=self.image_right), self.move_map_left(), self.update_ui(self.scorekeeper), self.get_next(data_fp, data_parser, self.scorekeeper) if not getattr(self, 'route_complete', False) else None],  # Squish Left
            lambda: [self.scorekeeper.save(self.image_left, route_position=self.movement_count, side='left', image_left=self.image_left, image_right=self.image_right), self.move_map_left(), self.update_ui(self.scorekeeper), self.get_next(data_fp, data_parser, self.scorekeeper) if not getattr(self, 'route_complete', False) else None]  # Save Left
        ]
        self.left_button_menu = LeftButtonMenu(self.root, [
            ("Inspect Left", self.left_action_callbacks[0]),
            ("Squish Left", self.left_action_callbacks[1]),
            ("Save Left", self.left_action_callbacks[2])
        ])
        self.right_action_callbacks = [
            lambda: [self.print_scenario_side_attributes('right'), self.scorekeeper.inspect(self.image_right, cost=get_inspect_cost_left_right(), route_position=self.movement_count, side='right'), self.update_ui(self.scorekeeper), self.check_game_end(data_fp, data_parser, self.scorekeeper)],  # Inspect Right
            lambda: [self.scorekeeper.squish(self.image_right, route_position=self.movement_count, side='right', image_left=self.image_left, image_right=self.image_right), self.move_map_right(), self.update_ui(self.scorekeeper), self.get_next(data_fp, data_parser, self.scorekeeper) if not getattr(self, 'route_complete', False) else None],  # Squish Right
            lambda: [self.scorekeeper.save(self.image_right, route_position=self.movement_count, side='right', image_left=self.image_left, image_right=self.image_right), self.move_map_right(), self.update_ui(self.scorekeeper), self.get_next(data_fp, data_parser, self.scorekeeper) if not getattr(self, 'route_complete', False) else None]  # Save Right
        ]
        self.right_button_menu = RightButtonMenu(self.root, [
            ("Inspect Right", self.right_action_callbacks[0]),
            ("Squish Right", self.right_action_callbacks[1]),
            ("Save Right", self.right_action_callbacks[2])
        ])
        if suggest:
            machine_buttons = [
                ("Suggest", lambda: [self.machine_interface.suggest(self.image_left)]),
                ("Act", lambda: [self.machine_interface.act(scorekeeper, self.image_left),
                                 self.update_ui(scorekeeper),
                                 self.get_next(data_fp, data_parser, scorekeeper) if not getattr(self, 'route_complete', False) else None])
            ]
            self.machine_menu = MachineMenu(self.root, machine_buttons)
        # Display two stacked (vertically) photos, centered horizontally
        image_width = 375
        horizontal_gap = 30  # pixels between images
        w, h = 1280, 800
        # Calculate total width for both images side by side
        total_width = image_width * 2
        # Center the pair of images
        center_x = (w - total_width) // 2
        y_top = 70 + 50 # Shift down by 50 pixels
        offset = 65 #shifting the images horizontally
        # Place left and right images side by side
        try:
            print("Creating game viewers (left and right)")
            self.game_viewer_left = GameViewer(self.root, image_width, h, data_fp, self.image_left)
            self.game_viewer_right = GameViewer(self.root, image_width, h, data_fp, self.image_right)
            print("Game viewers (left and right) created")
        except Exception as e:
            print("Exception creating game viewers (left and right):", e)
        # Place the canvases - left on the left, right on the right, both at y_top
        try:
            print("Placing game viewers (left and right)")
            self.game_viewer_left.canvas.place(x=center_x - offset, y=y_top)
            self.game_viewer_right.canvas.place(x=center_x - offset + image_width + horizontal_gap, y=y_top)
            print("Game viewers (left and right) placed")
        except Exception as e:
            print("Exception placing game viewers (left and right):", e)
            
        # Add imperfect CNN text labels above images
        try:
            # Get imperfect predictions for display
            left_image_path = os.path.join(self.data_fp, self.image_left.Filename)
            right_image_path = os.path.join(self.data_fp, self.image_right.Filename)
            
            left_text = self.imperfect_cnn.get_display_text(left_image_path, "LEFT")
            right_text = self.imperfect_cnn.get_display_text(right_image_path, "RIGHT")
            
            # Create text labels above images
            self.left_cnn_label = tk.Label(self.root, text=left_text, font=("Arial", 12), 
                                         bg="black", fg="yellow", wraplength=image_width-10)
            self.right_cnn_label = tk.Label(self.root, text=right_text, font=("Arial", 12), 
                                          bg="black", fg="yellow", wraplength=image_width-10)
            
            # Place labels above images
            self.left_cnn_label.place(x=center_x - offset, y=y_top - 40)
            self.right_cnn_label.place(x=center_x - offset + image_width + horizontal_gap, y=y_top - 40)
            print("Imperfect CNN labels placed")
        except Exception as e:
            print("Exception placing imperfect CNN labels:", e)
        self.root.bind("<Delete>", self.game_viewer_left.delete_photo)
        self.root.bind("<Delete>", self.game_viewer_right.delete_photo)
        # Display the countdown
        init_h = 12
        init_m = 0
        try:
            print("Creating clock")
            self.clock = Clock(self.root, w, h, init_h, init_m)
            print("Clock created")
        except Exception as e:
            print("Exception creating clock:", e)
        def get_ambulance_people():
            # Return a list of dicts from ambulance_people
            return list(self.scorekeeper.ambulance_people.values())
        try:
            print("Creating capacity meter")
            self.capacity_meter = CapacityMeter(self.root, w, h, capacity, get_ambulance_contents=get_ambulance_people)
            print("Capacity meter created")
        except Exception as e:
            print("Exception creating capacity meter:", e)
        rules_btn_width = 200
        rules_btn_x = 100
        rules_btn_y = 90
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
        self.grid_origin = (985, 495)  # location of map
        self.create_grid_map_canvas()  # Create the canvas first
        self.reset_map()               # Now it's safe to call reset_map()
        self.draw_grid_map()           # Draw the map with background image
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.direction_idx = 0  # Start facing right
        # self.create_grid_map_canvas() # This line is moved
        # self.draw_grid_map() # This line is moved

        # Movement progress label
        self.movement_label = tk.Label(self.root, text="Route Progress: 0/20", 
                                      font=("Arial", 12), bg="#000000", fg="#2E86AB",
                                      relief="solid", bd=1, padx=10, pady=5)
        self.movement_label.place(x=1050, y=460)

        # Initialize the UI with the current state
        self.update_ui(scorekeeper)
        self.time_warning_shown = False  # Track if the limited time warning popup has been shown

        # Restore to a single inspect_canvas as before
                # Add dual inspect canvases under each image
        inspect_height = 150
        canvas_gap_y = 10  # space between image and inspect canvas
        image_y = y_top
        image_height = self.game_viewer_left.canvas.winfo_reqheight()

        # Left canvas under left image
        self.inspect_canvas_left = tk.Canvas(self.root, width=image_width, height=inspect_height, bg="lightgreen", highlightthickness=0)
        self.inspect_canvas_left.place(x=center_x - offset, y=image_y + image_height + canvas_gap_y)
        self.inspect_canvas_left.create_rectangle(
        1, 1, image_width - 2, inspect_height - 2, outline="black", width=2
        )
        # Right canvas under right image
        self.inspect_canvas_right = tk.Canvas(self.root, width=image_width, height=inspect_height, bg="lightgreen", highlightthickness=0)
        self.inspect_canvas_right.place(x=center_x - offset + image_width + horizontal_gap, y=image_y + image_height + canvas_gap_y)
        # self.root.mainloop()  # Commented out - this was causing premature event loop start
        self.inspect_canvas_right.create_rectangle(
        1, 1, image_width - 2, inspect_height - 2, outline="black", width=2
        )
    
    def clear_main_ui(self):
        for widget in self.root.winfo_children():
            widget.destroy()
     
    def rebuild_main_ui(self):
    # Simply destroy the root window and restart the game
        self.root.destroy()
        # Create a new root window for the intro screen
        new_root = tk.Tk()
        IntroScreen(lambda: UI(self.data_parser, self.scorekeeper, self.data_fp, suggest=False, log=self.log, root=new_root), new_root)   
    
    def restore_main_ui(self):
        """Restore the main game UI without restarting the game"""
        # Store current state before clearing UI
        current_ambulance_pos = getattr(self, 'ambulance_pos', None)
        current_path_history = getattr(self, 'path_history', None)
        

        
        self.clear_main_ui()
        
        # Recreate the menu bar
        self.create_menu_bar()
        
        # Recreate the main game components
        w, h = 1280, 800
        self.root.configure(bg="black")
        self.root.title("Beaverworks SGAI 2025 - Team Splice")
        
        # Recreate button menu
        self.button_menu = ButtonMenu(self.root, self.user_buttons)
        
        # Recreate left/right button menus
        self.left_button_menu = LeftButtonMenu(self.root, [
            ("Inspect Left", self.left_action_callbacks[0]),
            ("Squish Left", self.left_action_callbacks[1]),
            ("Save Left", self.left_action_callbacks[2])
        ])
        
        self.right_button_menu = RightButtonMenu(self.root, [
            ("Inspect Right", self.right_action_callbacks[0]),
            ("Squish Right", self.right_action_callbacks[1]),
            ("Save Right", self.right_action_callbacks[2])
        ])
        
        # Recreate machine menu if it exists
        if hasattr(self, 'machine_interface'):
            machine_buttons = [
                ("Suggest", lambda: [self.machine_interface.suggest(self.image_left)]),
                ("Act", lambda: [self.machine_interface.act(self.scorekeeper, self.image_left),
                                 self.update_ui(self.scorekeeper),
                                 self.get_next(self.data_fp, self.data_parser, self.scorekeeper) if not getattr(self, 'route_complete', False) else None])
            ]
            self.machine_menu = MachineMenu(self.root, machine_buttons)
        
        # Recreate game viewers
        image_width = 375
        horizontal_gap = 30
        total_width = image_width * 2
        center_x = (w - total_width) // 2
        y_top = 70 + 50
        offset = 65
        
        self.game_viewer_left = GameViewer(self.root, image_width, h, self.data_fp, self.image_left)
        self.game_viewer_right = GameViewer(self.root, image_width, h, self.data_fp, self.image_right)
        self.game_viewer_left.canvas.place(x=center_x - offset, y=y_top)
        self.game_viewer_right.canvas.place(x=center_x - offset + image_width + horizontal_gap, y=y_top)
        
        # Recreate clock
        init_h = 12
        init_m = 0
        self.clock = Clock(self.root, w, h, init_h, init_m)
        
        # Recreate capacity meter
        def get_ambulance_people():
            return list(self.scorekeeper.ambulance_people.values())
        self.capacity_meter = CapacityMeter(self.root, w, h, self.scorekeeper.capacity, get_ambulance_contents=get_ambulance_people)
        
        # Recreate grid map
        self.create_grid_map_canvas()
        if current_ambulance_pos is not None and current_path_history is not None:
            # Restore ambulance position and path history
            self.ambulance_pos = current_ambulance_pos
            self.path_history = current_path_history
        else:
            # Fallback to reset if no stored state
            self.reset_map()
        self.draw_grid_map()
        
        # Recreate movement label
        self.movement_label = tk.Label(self.root, text=f"Route Progress: {self.movement_count}/20", 
                                      font=("Arial", 12), bg="#000000", fg="#2E86AB",
                                      relief="solid", bd=1, padx=10, pady=5)
        self.movement_label.place(x=1050, y=460)
        
        # Recreate inspect canvases
        inspect_height = 150
        canvas_gap_y = 10
        image_y = y_top
        image_height = self.game_viewer_left.canvas.winfo_reqheight()
        
        self.inspect_canvas_left = tk.Canvas(self.root, width=image_width, height=inspect_height, bg="lightgreen", highlightthickness=0)
        self.inspect_canvas_left.place(x=center_x - offset, y=image_y + image_height + canvas_gap_y)
        self.inspect_canvas_left.create_rectangle(
            1, 1, image_width - 2, inspect_height - 2, outline="black", width=2
        )
        
        self.inspect_canvas_right = tk.Canvas(self.root, width=image_width, height=inspect_height, bg="lightgreen", highlightthickness=0)
        self.inspect_canvas_right.place(x=center_x - offset + image_width + horizontal_gap, y=image_y + image_height + canvas_gap_y)
        self.inspect_canvas_right.create_rectangle(
            1, 1, image_width - 2, inspect_height - 2, outline="black", width=2
        )
        
        # Restore inspect canvas content based on scorekeeper's inspected state
        for side in ['left', 'right']:
            image = self.image_left if side == 'left' else self.image_right
            has_inspected_content = False
            for humanoid in image.humanoids:
                if humanoid is not None:
                    key = (side, humanoid.fp)
                    if self.scorekeeper.inspected_state.get(key, False):
                        has_inspected_content = True
                        break
            if has_inspected_content:
                self.print_scenario_side_attributes(side)
                # Disable the inspect button for this side since it's already been inspected
                if side == 'left':
                    self.left_button_menu.buttons[0].config(state='disabled')
                else:
                    self.right_button_menu.buttons[0].config(state='disabled')
        
        # Update the UI with current state
        self.update_ui(self.scorekeeper)

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
        self.clear_main_ui()
        # Set the root background to black
        self.root.configure(bg="black")
        tk.Label(self.root, text="Game Rules", font=("Arial", 24, "bold"), fg="white", bg="black").pack(pady=20)
        
        rules_text = (
            "Game Rules:\n"
            "- The goal is to complete the ambulance route (20 movements).\n"
            "- Choose an action: Skip if you want to skip this set of humanoids.\n"
            "- Inspect if you want to inspect the humanoids.\n"
            "- Squish if you want to squash the zombies.\n"
            "- Save if you want to save the humanoids.\n"
            "- Scram if your capacity is full.\n"
            "- The goal is to save as many humans and squash the zombies.\n"
            "- Saving humans gives you money for more upgrades when you scram.\n"
            "- Your choices affect your score.\n"
            "- Saving zombies or killing humans hurt your score, while saving humans and killing zombies adds to it."
            "- The game ends when you complete the route or the day ends. \n"
            "- If you don't complete your route on time, you lose points."
            "- Route progress is shown in the middle of your screen. \n"
            "- Remember: You can only save one set of humanoids, so pick wisely! \n"
            #More rules if needed (make sure to add \n)
            "- This is where we can add more rules in case we need to. \n\n"
        )

        tk.Label(self.root, text=rules_text, font=("Arial", 12), justify="left", padx=20, fg="white", bg="black").pack()

        tk.Button(self.root, text="Back to Game", font=("Arial", 14), fg="black", bg="white",
              command=self.restore_main_ui).pack(pady=30)

    def update_ui(self, scorekeeper):     
        # Use elapsed time to drive the clock forward
        elapsed_time = self.total_time - scorekeeper.remaining_time
        hours_elapsed = math.floor(elapsed_time / 60.0)
        minutes_elapsed = elapsed_time % 60
        # Clock starts at 8:00 AM
        start_hour = 8
        h = (start_hour + hours_elapsed) % 12
        if h == 0:
            h = 12
        m = minutes_elapsed
        print(f"I changed my time :> to {h} {m}")
        # Add force_pm flag if remaining_time <= 480
        force_pm = scorekeeper.remaining_time <= 480
        self.clock.update_time(h, m, force_pm=force_pm)
        self.capacity_meter.update_fill(scorekeeper.get_current_capacity())

        # Show a warning popup if 3 hours or less remain, but only once per game session
        if scorekeeper.remaining_time <= 180 and not getattr(self, 'time_warning_shown', False):
            self.time_warning_shown = True
            import tkinter.messagebox
            tkinter.messagebox.showwarning(
                "Limited Time Warning",
                "Warning: You have 3 hours or less remaining! Make your decisions carefully."
            )

    def on_resize(self, event):
        w = 0.6 * self.root.winfo_width()
        h = 0.7 * self.root.winfo_height()
        # Update both game viewers for dual screen setup
        self.game_viewer_left.canvas.config(width=w, height=h)
        self.game_viewer_right.canvas.config(width=w, height=h)

    def get_next(self, data_fp, data_parser, scorekeeper):
        remaining = len(data_parser.unvisited)
        remaining_time = scorekeeper.remaining_time
        # Ran out of humanoids or time? End game
        if remaining == 0 or remaining_time <= 0:
            self.update_ui(scorekeeper)  # Ensure clock and UI are updated before game end
            if self.log:
                # End-of-game: write log and increment run id
                scorekeeper.save_log(final=True)
            self.capacity_meter.update_fill(0)
            self.game_viewer_left.delete_photo(None)
            self.game_viewer_right.delete_photo(None)
            route_complete = getattr(self, 'movement_count', 0) >= 20 if hasattr(self, 'movement_count') else False
            final_score = scorekeeper.get_final_score(route_complete=route_complete)
            accuracy = round(scorekeeper.get_accuracy() * 100, 2)
            # Remove any previous final score frame
            if hasattr(self, 'final_score_frame') and self.final_score_frame:
                self.final_score_frame.destroy()
            # Create a new frame for the final score block
            self.final_score_frame = tk.Frame(self.root, width=300, height=300, bg="black", highlightthickness=0)
            self.final_score_frame.place(relx=0.5, rely=0.5, y=-100, anchor=tk.CENTER)  # Center in the whole window, shifted up 100px
            # 'Game Complete' label 
            if route_complete:
                final_score = self.scorekeeper.get_final_score(route_complete=True)
                game_complete_label = tk.Label(self.final_score_frame, text="Route Complete", font=("Arial", 40), fg="white", bg="black", highlightthickness=0)
                game_complete_label.pack(pady=(10, 5))
                final_score_label = tk.Label(self.final_score_frame, text="FINAL SCORE: " + f" {final_score}", font=("Arial", 16), fg="white", bg="black", highlightthickness=0)
                final_score_label.pack(pady=(5, 2))
                # Disable all left side buttons and hide the map on game end
                self.left_button_menu.disable_buttons(0, 0, True)
                self.right_button_menu.disable_buttons(0, 0, True)
                self.button_menu.disable_buttons(0, 0, True)
                # Force disable all buttons to ensure they stay disabled
                self.button_menu.force_disable_all_buttons()
                self.left_button_menu.force_disable_all_buttons()
                self.right_button_menu.force_disable_all_buttons()
                self.map_canvas.place_forget()
                self.draw_grid_map() # Redraw the map to hide it
            else:
                final_score = self.scorekeeper.get_final_score(route_complete=False)
                game_complete_label = tk.Label(self.final_score_frame, text="Game Complete", font=("Arial", 40), fg="white", bg="black", highlightthickness=0)
                game_complete_label.pack(pady=(10, 5))
                final_score_label = tk.Label(self.final_score_frame, text="FINAL SCORE: " + f" {final_score}", font=("Arial", 16), fg="white", bg="black", highlightthickness=0)
                final_score_label.pack(pady=(5, 2))
                # Disable all left side buttons and hide the map on game end
                self.left_button_menu.disable_buttons(0, 0, True)
                self.right_button_menu.disable_buttons(0, 0, True)
                self.button_menu.disable_buttons(0, 0, True)
                # Force disable all buttons to ensure they stay disabled
                self.button_menu.force_disable_all_buttons()
                self.left_button_menu.force_disable_all_buttons()
                self.right_button_menu.force_disable_all_buttons()
                self.map_canvas.place_forget()
                self.draw_grid_map() # Redraw the map to hide it
            # Scoring details
            killed_label = tk.Label(self.final_score_frame, text=f"Killed {scorekeeper.get_score(self.image_left, self.image_right)['killed']}", font=("Arial", 12), fg="white", bg="black", highlightthickness=0)
            killed_label.pack()
            saved_label = tk.Label(self.final_score_frame, text=f"Saved {scorekeeper.get_score(self.image_left, self.image_right)['saved']}", font=("Arial", 12), fg="white", bg="black", highlightthickness=0)
            saved_label.pack()
            
            zombie_ambu = tk.Label(self.final_score_frame, text=f"Zombies in Ambulance: {self.scorekeeper.false_saves}", font=("Arial", 12), fg="white", bg="black", highlightthickness=0)
            zombie_ambu.pack()

            # accuracy_label = tk.Label(self.final_score_frame, text=f"Accuracy: {accuracy:.2f}%", font=("Arial", 12), fg="white", bg="black", highlightthickness=0)
            # accuracy_label.pack()
            zombies_saved_score_label = tk.Label(self.final_score_frame,
                text=f"Zombies Saved Score: {self.scorekeeper.false_saves}",
                font=("Arial", 12),
                fg="white",
                bg="black",
                highlightthickness=0)
            zombies_saved_score_label.pack()

            # Replay button
            self.replay_btn = tk.Button(self.final_score_frame, text="Replay", command=lambda: self.reset_game(data_parser, data_fp))
            self.replay_btn.pack(pady=(10, 0))
            self.replay_btn.lift()  # Ensure the replay button is visible on top
            # Disable all left side buttons and hide the map on game end
            self.left_button_menu.disable_buttons(0, 0, True)
            self.right_button_menu.disable_buttons(0, 0, True)
            self.button_menu.disable_buttons(0, 0, True)
            # Force disable all buttons to ensure they stay disabled
            self.button_menu.force_disable_all_buttons()
            self.left_button_menu.force_disable_all_buttons()
            self.right_button_menu.force_disable_all_buttons()
            self.map_canvas.place_forget()
            self.draw_grid_map() # Redraw the map to hide it
        else:
            # Only load new images if route is not complete
            if not self.route_complete:
                self.image_left, self.image_right = data_parser.get_random(side='left'), data_parser.get_random(side='right')
                # Reset inspection state for new scenario
                self.scorekeeper.inspected_left = False
                self.scorekeeper.inspected_right = False
  
                # self.humanoid_left, self.humanoid_right, scenario_number, scenario_desc, self.scenario_humanoid_attributes = data_parser.get_scenario()
                # print(f"[UI DEBUG] Scenario {scenario_number}: left={scenario_desc[0]}, right={scenario_desc[1]}")
                # Clear inspect canvas text
                self.inspect_canvas_left.delete('all')
                self.inspect_canvas_right.delete('all')
                # Process zombie infections at the start of each turn
                infected_humanoids = scorekeeper.process_zombie_infections()
                if infected_humanoids:
                    # Debug prints
                    print(f"[ZOMBIE INFECTION] The following humanoids were turned into zombies:")
                    for humanoid_id in infected_humanoids:
                        print(f"[ZOMBIE INFECTION] {humanoid_id} was turned into a zombie!")
                    # Create popup message for zombie infections
                    infection_message = "ZOMBIE INFECTION!\n\nThe following humanoids were turned into zombies:\n"
                    for humanoid_id in infected_humanoids:
                        infection_message += f"• {humanoid_id}\n"
                    infection_message += "\nThe ambulance is now more dangerous!"
                    # Show popup
                    _safe_show_popup("Zombie Infection!", infection_message, 'warning')

                # Process zombie cures at the start of each turn
                cured_humanoids = scorekeeper.process_zombie_cures()
                if cured_humanoids:
                    print(f"[CURE] The following zombies were cured:")
                    for humanoid_id in cured_humanoids:
                        print(f"[CURE] {humanoid_id} was cured and is now a human civilian!")
                    cure_message = "CURE!\n\nThe following zombies were cured and are now human civilians:\n"
                    for humanoid_id in cured_humanoids:
                        cure_message += f"• {humanoid_id}\n"
                    cure_message += "\nDoctors have made the ambulance safer!"
                    _safe_show_popup("Zombie Cure!", cure_message)
                fp_left = os.path.join(data_fp, self.image_left.Filename)
                fp_right = os.path.join(data_fp, self.image_right.Filename)
                self.game_viewer_left.create_photo(fp_left)
                self.game_viewer_right.create_photo(fp_right)
                
                # Update imperfect CNN labels for new images
                try:
                    if hasattr(self, 'left_cnn_label') and hasattr(self, 'right_cnn_label'):
                        # Use correct image paths by joining with data_fp
                        left_image_path = os.path.join(self.data_fp, self.image_left.Filename)
                        right_image_path = os.path.join(self.data_fp, self.image_right.Filename)
                        
                        left_text = self.imperfect_cnn.get_display_text(left_image_path, "LEFT")
                        right_text = self.imperfect_cnn.get_display_text(right_image_path, "RIGHT")
                        
                        self.left_cnn_label.config(text=left_text)
                        self.right_cnn_label.config(text=right_text)
                        print("Imperfect CNN labels updated for new images")
                    else:
                        print("CNN labels not found, skipping update")
                except Exception as e:
                    print("Exception updating imperfect CNN labels:", e)

        # Disable buttons that would exceed time limit
        if remaining > 0 and remaining_time > 0:
            self.disable_buttons_if_insufficient_time(remaining_time, remaining, scorekeeper.at_capacity())

    def check_game_end(self, data_fp, data_parser, scorekeeper):
        """Check if game should end due to time running out"""
        remaining_time = scorekeeper.remaining_time
        if remaining_time <= 0:
            # Disable all buttons when time runs out
            self.button_menu.disable_buttons(0, 0, True, 0, 0, 1, 1)
            self.left_button_menu.disable_buttons(0, 0, True, 0, 0, 1, 1)
            self.right_button_menu.disable_buttons(0, 0, True, 0, 0, 1, 1)
            if hasattr(self, 'machine_menu'):
                self.machine_menu.disable_buttons(0, 0, True)
            # Safely disable menu buttons if they exist
            if hasattr(self, 'rules_btn'):
                self.rules_btn.config(state='disabled')
            if hasattr(self, 'upgrade_btn'):
                self.upgrade_btn.config(state='disabled')
            self.trigger_end_screen()

            # Upload data and clear log
            self.upload_data_and_clear_log()

    def upload_data_and_clear_log(self):
        """Upload data to Notion and clear the log file"""
        try:
            print("Uploading data to Notion...")
            # Use the DataUploader class to upload data
            success = self.data_uploader.upload_data()
            
            if success:
                print("Data uploaded successfully!")
            else:
                print("Upload failed")
        except Exception as e:
            print(f"Error uploading data: {e}")
        
        # Clear the log file after upload (preserve headers)
        try:
            if os.path.exists("log.csv"):
                # Read the first line (headers) and write it back
                with open("log.csv", "r") as f:
                    first_line = f.readline().strip()
                with open("log.csv", "w") as f:
                    f.write(first_line + "\n")  # Write back only the headers
                print("log.csv cleared after data upload (headers preserved)")
        except Exception as e:
            print(f"Error clearing log.csv: {e}")

    def disable_buttons_if_insufficient_time(self, remaining_time, remaining_humanoids, at_capacity):
        """Disable buttons based on remaining time and other constraints"""
        # Don't re-enable buttons if the game is complete
        if hasattr(self, 'route_complete') and self.route_complete:
            return
            
        # Get humanoid counts for capacity checking
        left_humanoid_count = getattr(self.image_left, 'datarow', {}).get('HumanoidCount', 1) if hasattr(self, 'image_left') and self.image_left else 1
        right_humanoid_count = getattr(self.image_right, 'datarow', {}).get('HumanoidCount', 1) if hasattr(self, 'image_right') and self.image_right else 1
        current_capacity = len(self.scorekeeper.ambulance_people)
        ambulance_capacity = self.scorekeeper.capacity
        
        # Use the existing ButtonMenu.disable_buttons method
        # Note: ButtonMenu now handles Skip (index 0), Inspect (index 1), Squish (index 2), Save (index 3)
        self.button_menu.disable_buttons(remaining_time, remaining_humanoids, at_capacity, 
                                       current_capacity, ambulance_capacity, left_humanoid_count, right_humanoid_count)
        
        # Also disable the left and right button menus with the same logic
        self.left_button_menu.disable_buttons(remaining_time, remaining_humanoids, at_capacity,
                                            current_capacity, ambulance_capacity, left_humanoid_count, right_humanoid_count)
        self.right_button_menu.disable_buttons(remaining_time, remaining_humanoids, at_capacity,
                                             current_capacity, ambulance_capacity, left_humanoid_count, right_humanoid_count)
        
            
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
        self.movement_count = 0  # Reset movement counter
        self.false_saves = 0
        self.route_complete = False  # Reset route complete flag
        self.movement_label.config(text="Route Progress: 0/20")  # Reset movement label
        self.scorekeeper.reset()
        data_parser.reset()
        # 1. Fully reset the clock (including blink, time, force_pm)
        self.clock.blink = True
        self.clock.current_h = 8
        self.clock.current_m = 0
        self.clock.force_pm = False
        self.clock.update_time(8, 0, force_pm=False)
        # 2. Reset ambulance map and position
        self.reset_map() # This sets ambulance_pos and path_history to start
        self.hide_map = False
        self.draw_grid_map()
        self.map_canvas.place(x=self.grid_origin[0], y=self.grid_origin[1]) # Always re-place the map after replay
        # 3. Re-enable all buttons
        self.button_menu.enable_all_buttons()
        self.left_button_menu.enable_all_buttons()
        self.right_button_menu.enable_all_buttons()
        if hasattr(self, 'machine_menu'):
            self.machine_menu.enable_all_buttons()
        # Safely enable menu buttons if they exist
        if hasattr(self, 'rules_btn'):
            self.rules_btn.config(state='normal')
        if hasattr(self, 'upgrade_btn'):
            self.upgrade_btn.config(state='normal')
        # 4. Get a new scenario (ensure new images are generated)
        self.image_left, self.image_right = data_parser.get_random(side='left'), data_parser.get_random(side='right')
        #TODO: fix to be getting images, not humanoids
        # self.humanoid_left, self.humanoid_right, scenario_number, scenario_desc = data_parser.get_scenario()
        # print(f"[UI DEBUG] Reset Scenario {scenario_number}: left={scenario_desc[0]}, right={scenario_desc[1]}")
        fp_left = os.path.join(data_fp, self.image_left.Filename)
        fp_right = os.path.join(data_fp, self.image_right.Filename)
        self.game_viewer_left.create_photo(fp_left)
        self.game_viewer_right.create_photo(fp_right)
        # 5. Reset scram cost to starting value
        if hasattr(self.scorekeeper, 'scram_time_reduction'):
            self.scorekeeper.scram_time_reduction = 0
        self.button_menu.buttons[4].config(text="Scram (5 mins)")
        self.update_ui(self.scorekeeper)
        # Clear any widgets in both canvases
        for widget in self.game_viewer_left.canvas.pack_slaves():
            widget.destroy()
        for widget in self.game_viewer_right.canvas.pack_slaves():
            widget.destroy()
        # Clear inspect canvas
        self.inspect_canvas_left.delete('all')
        self.inspect_canvas_right.delete('all')
        self.time_warning_shown = False  # Reset the warning flag for a new game
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
            # Increment movement counter
            self.movement_count += 1
            # print(f"[DEBUG] Ambulance movement {self.movement_count}/20")
            
            # Update movement progress label
            self.movement_label.config(text=f"Route Progress: {self.movement_count}/20")
            
            # Check if we've reached 20 movements
            if self.movement_count >= 20:
                # print("[DEBUG] Reached 20 movements - triggering end screen")
                self.trigger_end_screen()
        self.draw_grid_map()

    def trigger_end_screen(self):
        """Trigger the end screen when 20 movements are reached"""
        self.route_complete = True  # Set flag to prevent new images from loading
        self.update_ui(self.scorekeeper)  # Ensure clock and UI are updated before end screen
        if self.log:
            # End-of-game: write log and increment run id
            self.scorekeeper.save_log(final=True)
        self.capacity_meter.update_fill(0)
        self.game_viewer_left.delete_photo(None)
        self.game_viewer_right.delete_photo(None)
        final_score = self.scorekeeper.get_final_score(route_complete=True)
        accuracy = round(self.scorekeeper.get_accuracy() * 100, 2)
        # Remove any previous final score frame
        if hasattr(self, 'final_score_frame') and self.final_score_frame:
            self.final_score_frame.destroy()
        # Create a new frame for the final score block
        self.final_score_frame = tk.Frame(self.root, width=300, height=300,bg="black",highlightthickness=0)
        self.final_score_frame.place(relx=0.5, rely=0.5, y=-100, anchor=tk.CENTER)  # Center in the whole window, shifted up 100px
        # 'Route Complete' label
        game_complete_label = tk.Label(self.final_score_frame, text="Route Complete", font=("Arial", 40),fg="white",bg="black",highlightthickness=0)
        game_complete_label.pack(pady=(10, 5))
        # 'Final Score' label
        final_score_label = tk.Label(self.final_score_frame, text=f"FINAL SCORE: {final_score}", font=("Arial", 16),fg="white",bg="black",highlightthickness=0)
        final_score_label.pack(pady=(5, 2))
        # Scoring details
        killed_label = tk.Label(self.final_score_frame, text=f"Killed {self.scorekeeper.get_score(self.image_left, self.image_right)['killed']}", font=("Arial", 12),fg="white",bg="black",highlightthickness=0)
        killed_label.pack()
        saved_label = tk.Label(self.final_score_frame, text=f"Saved {self.scorekeeper.get_score(self.image_left, self.image_right)['saved']}", font=("Arial", 12),fg="white",bg="black",highlightthickness=0)
        saved_label.pack()
        zombie_ambu = tk.Label(self.final_score_frame, text=f"Zombies in Ambulance: {self.scorekeeper.false_saves}", font=("Arial", 12), fg="white", bg="black", highlightthickness=0)
        zombie_ambu.pack()

        zombies_saved_score_label = tk.Label(self.final_score_frame,
            text=f"Zombies Saved Score: {self.scorekeeper.false_saves}",
            font=("Arial", 12),
            fg="white",
            bg="black",
            highlightthickness=0)
        zombies_saved_score_label.pack()

        # accuracy_label = tk.Label(
        #     self.final_score_frame,
        #     text=f"Accuracy: {accuracy:.2f}%",
        #     font=("Arial", 12),
        #     fg="white",
        #     bg="black",
        #     highlightthickness=0
        # )
        # accuracy_label.pack()
        # Replay button (only one)
        self.replay_btn = tk.Button(self.final_score_frame, text="Replay", command=lambda: self.reset_game(self.data_parser, self.data_fp))
        self.replay_btn.pack(pady=(10, 0))
        self.replay_btn.lift()  # Ensure the replay button is visible on top
        self.replay_btn.config(state='normal')
        # Disable all left, right, and main button menus and hide the map on game end
        self.left_button_menu.disable_buttons(0, 0, True)
        self.right_button_menu.disable_buttons(0, 0, True)
        self.button_menu.disable_buttons(0, 0, True)
        self.map_canvas.place_forget()  # Hide the map
        # Remove any content from the canvasses
        self.game_viewer_left.delete_photo(None)
        self.game_viewer_right.delete_photo(None)
        #self.game_viewer_right.canvas.delete('all') # not sure if necessary
        # Disable all button functionality when route is complete
        self.button_menu.force_disable_all_buttons()
        self.left_button_menu.force_disable_all_buttons()
        self.right_button_menu.force_disable_all_buttons()
        if hasattr(self, 'machine_menu'):
            self.machine_menu.disable_buttons(0, 0, True)
        # Safely disable menu buttons if they exist
        if hasattr(self, 'rules_btn'):
            self.rules_btn.config(state='disabled')
        if hasattr(self, 'upgrade_btn'):
            self.upgrade_btn.config(state='disabled')
        
        # Upload data and clear log
        self.upload_data_and_clear_log()

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
        self.clear_main_ui()
        # Set the root background to black
        self.root.configure(bg="black")

    # Title
        tk.Label(self.root, text="Upgrade Shop", font=("Arial", 24, "bold"), fg="white", bg="black").pack(pady=20)

    # Money display
        money = self.scorekeeper.upgrade_manager.get_money()
        money_label = tk.Label(self.root, text=f"Money: ${money}", font=("Arial", 16), fg="white", bg="black")
        money_label.pack(pady=10)

    # Each upgrade
        colors = ["#ff6666", "#66ff66", "#6666ff"]  # Red, Green, Blue
        color_index = 0
        
        for name, info in self.scorekeeper.upgrade_manager.upgrades.items():
            upgrade_label = name.replace("_", " ").title()
            level = info["level"]
            cost = info["cost"]

            def make_purchase(n=name):
                print(f"Attempting to purchase: {n}")
                if self.scorekeeper.upgrade_manager.purchase(n):
                    print(f"Purchase successful: {n}")
                    # Force immediate UI update and refresh for all upgrades
                    self.root.update_idletasks()
                    self.show_upgrade_shop()  # Refresh the shop view
                else:
                    print(f"Purchase failed: {n}")

            btn_text = f"{upgrade_label} (Level {level})"
            if level >= self.scorekeeper.upgrade_manager.upgrades[name]["max"]:
                btn_text += " (MAX)"
                btn = tk.Button(self.root, text=btn_text, font=("Arial", 12),
                            state='disabled', bg="#dddddd", disabledforeground="gray")
            else:
                btn_text += f" - ${cost}"
                btn = tk.Button(self.root, text=btn_text, font=("Arial", 12), command=make_purchase, 
                               fg="black", bg=colors[color_index % len(colors)])

            btn.pack(pady=5)
            color_index += 1

    # Back button
        tk.Button(self.root, text="Back to Game", font=("Arial", 14), fg="black", bg="white",
                command=self.restore_main_ui).pack(pady=30)

    def print_scenario_side_attributes(self, side):

        if side == 'left':
            image = self.image_left
        elif side == 'right':
            image = self.image_right
        else:
            print(f"Unknown side: {side}")
            return

        lines = [f"{side.title()} side:"]
        print(f"Attributes for {side} image:")
        for idx, humanoid in enumerate(image.humanoids):
            if humanoid is not None:
                lines.append(f"  Humanoid {idx + 1}:")
                #lines.append(f"    File path: {humanoid.fp}")
                lines.append(f"    State: {humanoid.state}")
                # Always display the original role, regardless of state
                lines.append(f"    Role: {humanoid.role}")
                # Add more attributes if you want, e.g. gender, item, etc.
            else:
                pass
                # lines.append(f"  Humanoid {idx}: None")


        # for i in range(1, 4):
        #     key = f"{side}_humanoid{i}"
        #     attrs = self.scenario_humanoid_attributes.get(key, {})
        #     if attrs and attrs.get('type', '').strip():
        #         lines.append(f"{key}:")
        #         lines.append(f"  Type: {attrs['type']}")
        #         lines.append(f"  Status: {attrs['status']}")
        #         lines.append(f"  Role: {attrs['role']}")
        #         lines.append("")  # Blank line between humanoids
        text = '\n'.join(lines)
        # # # Display scenario attributes in the appropriate inspect canvas (left or right)
        if side == 'left':
            canvas = self.inspect_canvas_left
            self.left_button_menu.buttons[0].config(state='disabled')
        else:
            canvas = self.inspect_canvas_right
            self.right_button_menu.buttons[0].config(state='disabled')

        canvas.delete('all')
        canvas.create_rectangle(1, 1, canvas.winfo_reqwidth() - 2, canvas.winfo_reqheight() - 2, outline="black", width=2)
        canvas.create_text(10, 10, anchor='nw', text=text, font=("Arial", 12))

    def add_elapsed_time(self, minutes):
        self.scorekeeper.remaining_time -= minutes

    def create_menu_bar(self):
        self.menu_bar = tk.Frame(self.root, bg="#000000", height=50)
        self.menu_bar.place(x=0, y=0, width=1280, height=50)

    # Styled buttons
        btn_font = ("Helvetica", 14, "bold")

        self.upgrade_btn = tk.Button(self.menu_bar, text="Upgrades", font=btn_font, bg="#ffffff", fg="#2E86AB",
              relief="raised", padx=10, pady=5, command=self.show_upgrade_shop)
        self.upgrade_btn.pack(side="left", padx=1)

        self.rules_btn = tk.Button(self.menu_bar, text="Rules", font=btn_font, bg="#ffffff", fg="#2E86AB",
              relief="raised", padx=10, pady=5, command=self.show_rules)
        self.rules_btn.pack(side="left", padx=1)

        tk.Button(self.menu_bar, text="Exit", font=btn_font, bg="#ffffff", fg="#AA0000",
              relief="raised", padx=10, pady=5, command=self.root.quit).pack(side="left", padx=1)
