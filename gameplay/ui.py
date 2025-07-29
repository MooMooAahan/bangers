import math
import tkinter as tk
from tkinter import font
from ui_elements.button_menu import ButtonMenu, LeftButtonMenu, RightButtonMenu
from ui_elements.capacity_meter import CapacityMeter
from ui_elements.clock import Clock
from ui_elements.ambulance_overlay import AmbulanceOverlay
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
        self.root.title("Welcome - Team Splice (SGAI 2025)")
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
        # Create a canvas to hold the background image and UI elements
        self.canvas = tk.Canvas(self.root, width=600, height=350, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Load and resize the background image
        try:
            # Try to load the logo image as background
            image_path = "images/intro_image.png"
            original_image = Image.open(image_path)
            
            # Resize image to fit the window (600x350)
            resized_image = original_image.resize((600, 350), Image.Resampling.LANCZOS)
            self.bg_image = ImageTk.PhotoImage(resized_image)
            
            # Place the background image on the canvas
            self.canvas.create_image(0, 0, image=self.bg_image, anchor="nw")
            
            # Add a semi-transparent overlay to make text more readable
            # self.canvas.create_rectangle(0, 0, 600, 350, fill="black", stipple="gray50")
            
        except Exception as e:
            print(f"Could not load background image: {e}")
            # Fallback to a solid color background
            self.canvas.configure(bg="darkblue")
        
        # Import theme fonts
        from ui_elements.theme import UPGRADE_FONT, TITLE_FONT_FOR_CURRENT_TIME
        
        # Add the title text directly on the canvas (truly transparent)
        self.canvas.create_text(300, 60, text="Team Splice (SGAI 2025)",
                               font=TITLE_FONT_FOR_CURRENT_TIME, 
                               fill="white", anchor="center")
        
        # Add the start button directly on the canvas (positioned to the left)
        self.start_button = tk.Button(self.canvas, text="Start Game", 
                                    font=UPGRADE_FONT, 
                                    width=12, height=1,
                                    bg="#5B7B7A", fg="white",  # Green background, white text
                                    relief="raised", bd=3,
                                    command=self.start_game)
        self.canvas.create_window(200, 105, window=self.start_button, anchor="center")
        
        # Add the instructions button directly on the canvas (positioned to the right)
        self.instructions_button = tk.Button(self.canvas, text="Instructions", 
                                           font=UPGRADE_FONT, 
                                           width=12, height=1,
                                           bg="#5B7B7A", fg="white",  # Green background, white text
                                           relief="raised", bd=3,
                                           command=self.show_instructions)
        self.canvas.create_window(400, 105, window=self.instructions_button, anchor="center")
    def start_game(self):
        # Change button appearance to pressed state
        self.start_button.config(
            bg="#4A6B6A",  # Darker green for pressed state
            relief="sunken",  # Sunken relief to show pressed
            state="disabled"  # Disable the button
        )
        
        # Show loading text
        self.show_loading()
        
        # Don't destroy the intro screen here - let the callback handle it
        self.on_start_callback()
    
    def show_loading(self):
        # Add loading text at the bottom of the screen
        from ui_elements.theme import UPGRADE_FONT
        self.loading_text = self.canvas.create_text(300, 325, text="Loading...",
                                                   font=UPGRADE_FONT, 
                                                   fill="white", anchor="center")
        
        # Force update to show the loading text immediately
        self.root.update_idletasks()
    
    def show_instructions(self):
        """Show game instructions in a popup dialog"""
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Game Instructions - Team Splice")
        popup.geometry("800x600")
        popup.resizable(False, False)
        
        # Center the popup on the screen
        popup.transient(self.root)
        popup.grab_set()
        
        # Create a canvas to hold the background image and UI elements
        canvas = tk.Canvas(popup, width=800, height=600, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        
        # Load and resize the city background image
        try:
            image_path = "images/city_background.png"
            original_image = Image.open(image_path)
            
            # Resize image to fit the window (800x600)
            resized_image = original_image.resize((800, 600), Image.Resampling.LANCZOS)
            self.bg_image_instructions = ImageTk.PhotoImage(resized_image)  # Store reference
            
            # Place the background image on the canvas
            canvas.create_image(0, 0, image=self.bg_image_instructions, anchor="nw")
            
            # Add a semi-transparent overlay to make text more readable
            canvas.create_rectangle(0, 0, 800, 600, fill="black", stipple="gray50")
            
        except Exception as e:
            print(f"Could not load city background image: {e}")
            # Fallback to a solid color background
            canvas.configure(bg="darkblue")
        
        # Title label with start screen styling
        from ui_elements.theme import TITLE_FONT_FOR_CURRENT_TIME
        canvas.create_text(400, 40, text="Game Instructions", 
                          font=TITLE_FONT_FOR_CURRENT_TIME, 
                          fill="white", anchor="center")
        
        # Instructions text - using the same comprehensive text as show_rules
        instructions_text = (
            "Welcome to Team Splice (SGAI 2025)!\n\n"
            
            "🎯 OBJECTIVE:\n"
            "Complete the ambulance route (20 movements) while saving humans and eliminating zombies.\n\n"
            
            "🎮 GAMEPLAY:\n"
            "• You'll encounter pairs of humanoids (humans and zombies)\n"
            "• Choose your actions carefully - you can only save one set per scenario\n"
            "• Complete your route before time runs out\n\n"
            
            "🔧 ACTIONS:\n"
            "• SKIP: Pass on the current humanoids\n"
            "• INSPECT: Get more information (costs time)\n"
            "• SQUISH: Eliminate zombies (costs time)\n"
            "• SAVE: Rescue humans (costs time and capacity)\n"
            "• SCRAM: Return to base when capacity is full\n\n"
            
            "📊 SCORING:\n"
            "• +Points: Saving humans, eliminating zombies\n"
            "• -Points: Saving zombies, eliminating humans\n"
            "• Money earned from saving humans can be used for upgrades\n\n"
            
            "⏰ TIME MANAGEMENT:\n"
            "• You have 12 hours (720 minutes) to complete your route\n"
            "• Route progress is shown in the center of your screen\n"
            "• If you don't complete the route on time, you lose points\n\n"
            
            "💡 TIPS:\n"
            "• Use INSPECT to gather information before making decisions\n"
            "• Manage your capacity wisely - you can't save everyone\n"
            "• Watch the clock and plan your route completion\n"
            "• Remember: You can only save one set per scenario!"
        )
        
        # Create a scrollable text widget for instructions
        text_frame = tk.Frame(canvas, bg="#1a1a1a")
        text_frame.place(relx=0.5, rely=0.5, anchor="center", width=700, height=400)
        
        # Create text widget with scrollbar
        from ui_elements.theme import LABEL_FONT, UPGRADE_FONT
        text_widget = tk.Text(text_frame, wrap="word", font=LABEL_FONT, 
                             bg="#1a1a1a", fg="white", relief="flat",
                             padx=20, pady=20, state="normal")
        scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Pack the text widget and scrollbar
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Insert the instructions text
        text_widget.insert("1.0", instructions_text)
        text_widget.config(state="disabled")  # Make read-only
        
        # Add a close button with start screen styling
        from ui_elements.theme import UPGRADE_FONT
        close_button = tk.Button(canvas, text="Close", 
                               font=UPGRADE_FONT, 
                               width=12, height=1,
                               bg="#5B7B7A", fg="white",  # Green background, white text
                               relief="raised", bd=3,
                               command=popup.destroy)
        canvas.create_window(400, 550, window=close_button, anchor="center")
    
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
        self.root.title("Beaverworks SGAI 2025 - Team Splice")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (w // 2)
        y = (screen_height // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.resizable(False, False)
        
        # Create ambulance view as underlay FIRST
        # try:
        #     print("Creating ambulance view")
        #     self.ambulance_overlay = AmbulanceOverlay(self.root, w, h)
        #     self.ambulance_overlay.place(0, 0)  # Place at the very bottom layer
        #     print("Ambulance view created and placed")
        # except Exception as e:
        #     print("Exception creating ambulance view:", e)
        
        # Create menu bar AFTER overlay so it appears on top
        self.create_menu_bar()
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
            (get_scram_text(), lambda: [
                                    # print(f"[DEBUG] Scram penalty applied: {get_scram_time()} minutes"),
                                    scorekeeper.scram(self.image_left, self.image_right, time_cost=get_scram_time(), route_position=self.movement_count),
                                    self.move_ambulance_by_cell() if not self.route_complete else None,
                                    self.update_ui(scorekeeper),
                                    self.get_next(
                                        data_fp,
                                        data_parser,
                                        scorekeeper) if not getattr(self, 'route_complete', False) else None],COLOR_SCRAM),
            (f"Inspect ({get_inspect_cost()} mins)", lambda: [self.show_action_popup("Inspect")],COLOR_INSPECT),
            ("Squish (5 mins)", lambda: self.show_action_popup("Squish"),COLOR_SQUISH),
            ("Save (30 mins)", lambda: self.show_action_popup("Save"),COLOR_SAVE),
            ("Skip (15 mins)", lambda: [
                                  scorekeeper.skip_both(self.image_left, self.image_right, route_position=self.movement_count),
                                  self.move_ambulance_by_cell() if not self.route_complete else None,
                                  self.update_ui(scorekeeper),
                                  self.get_next(
                                      data_fp,
                                      data_parser,
                                      scorekeeper) if not getattr(self, 'route_complete', False) else None],COLOR_SKIP)
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
            # Button order: Scram, Inspect, Squish, Save, Skip
            if hasattr(self, 'button_menu') and self.button_menu:
                self.button_menu.buttons[0].config(text=get_scram_text())  # Scram
                self.button_menu.buttons[1].config(text=f"Inspect ({get_inspect_cost()} mins)")  # Inspect
                self.button_menu.buttons[2].config(text="Squish (5 mins)")  # Squish
                self.button_menu.buttons[3].config(text="Save (30 mins)")  # Save
                self.button_menu.buttons[4].config(text="Skip (15 mins)")  # Skip
        self.update_button_texts = update_button_texts

        # Call update_button_texts after every UI update
        orig_update_ui = self.update_ui
        def patched_update_ui(scorekeeper):
            orig_update_ui(scorekeeper)
            if hasattr(self, 'update_button_texts'):
                self.update_button_texts()
        self.update_ui = patched_update_ui

        # Restore left/right button menus for Squish/Save
        # Add extra button menu with three buttons
        # Save references to left/right button actions for map movement
        self.left_action_callbacks = [
            lambda: [self.print_scenario_side_attributes('left'), self.scorekeeper.inspect(self.image_left, cost=get_inspect_cost_left_right(), route_position=self.movement_count, side='left'), self.update_ui(self.scorekeeper), self.check_game_end(data_fp, data_parser, self.scorekeeper)],  # Inspect Left
            lambda: [self.scorekeeper.squish(self.image_left, route_position=self.movement_count, side='left', image_left=self.image_left, image_right=self.image_right), self.move_map_left() if not self.route_complete else None, self.update_ui(self.scorekeeper), self.get_next(data_fp, data_parser, self.scorekeeper) if not getattr(self, 'route_complete', False) else None],  # Squish Left
            lambda: [self.scorekeeper.save(self.image_left, route_position=self.movement_count, side='left', image_left=self.image_left, image_right=self.image_right), self.move_map_left() if not self.route_complete else None, self.update_ui(self.scorekeeper), self.get_next(data_fp, data_parser, self.scorekeeper) if not getattr(self, 'route_complete', False) else None]  # Save Left
        ]
        try:
            print("Creating left/right button menus")
            self.left_button_menu = LeftButtonMenu(self.root, [
                ("Inspect Left", self.left_action_callbacks[0]),
                ("Squish Left", self.left_action_callbacks[1]),
                ("Save Left", self.left_action_callbacks[2])
            ])
            self.right_action_callbacks = [
                lambda: [self.print_scenario_side_attributes('right'), self.scorekeeper.inspect(self.image_right, cost=get_inspect_cost_left_right(), route_position=self.movement_count, side='right'), self.update_ui(self.scorekeeper), self.check_game_end(data_fp, data_parser, self.scorekeeper)],  # Inspect Right
                lambda: [self.scorekeeper.squish(self.image_right, route_position=self.movement_count, side='right', image_left=self.image_left, image_right=self.image_right), self.move_map_right() if not self.route_complete else None, self.update_ui(self.scorekeeper), self.get_next(data_fp, data_parser, self.scorekeeper) if not getattr(self, 'route_complete', False) else None],  # Squish Right
                lambda: [self.scorekeeper.save(self.image_right, route_position=self.movement_count, side='right', image_left=self.image_left, image_right=self.image_right), self.move_map_right() if not self.route_complete else None, self.update_ui(self.scorekeeper), self.get_next(data_fp, data_parser, self.scorekeeper) if not getattr(self, 'route_complete', False) else None]  # Save Right
            ]
            self.right_button_menu = RightButtonMenu(self.root, [
                ("Inspect Right", self.right_action_callbacks[0]),
                ("Squish Right", self.right_action_callbacks[1]),
                ("Save Right", self.right_action_callbacks[2])
            ])
            print("Left/right button menus created")
        except Exception as e:
            print("Exception creating left/right button menus:", e)
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
            from ui_elements.theme import LABEL_FONT
            self.left_cnn_label = tk.Label(self.root, text=left_text, font=LABEL_FONT, 
                                         bg="black", fg="yellow", wraplength=image_width-10)
            self.right_cnn_label = tk.Label(self.root, text=right_text, font=LABEL_FONT, 
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
        init_h = 8
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
        self.cell_size = 35  # Small for better fit
        self.grid_origin = (1040, 500)  # location of map (centered with clock and capacity meter)
        self.create_grid_map_canvas()  # Create the canvas first
        self.reset_map()               # Now it's safe to call reset_map()
        self.draw_grid_map()           # Draw the map with background image
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.direction_idx = 0  # Start facing right
        # self.create_grid_map_canvas() # This line is moved
        # self.draw_grid_map() # This line is moved

        # Movement progress label
        movement_font = font.Font(family="Fixedsys", size=10)
        self.movement_label = tk.Label(self.root, text="Route Progress: 0/20", 
                                      font=movement_font, bg="#5B7B7A", fg="#FFFFFF",
                                      relief="groove", bd=2, padx=10, pady=5)
        self.movement_label.place(x=1060, y=460)

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
    # Instead of destroying and recreating the root window, clear and reuse it
        self.clear_main_ui()
        IntroScreen(lambda: UI(self.data_parser, self.scorekeeper, self.data_fp, suggest=False, log=self.log, root=self.root), self.root)
    
    def restore_main_ui(self):
        """Restore the main game UI without restarting the game"""
        # Store current state before clearing UI
        current_ambulance_pos = getattr(self, 'ambulance_pos', None)
        current_path_history = getattr(self, 'path_history', None)
        

        
        self.clear_main_ui()
        
        # Recreate the main game components
        w, h = 1280, 800
        self.root.configure(bg="black")
        self.root.title("Beaverworks SGAI 2025 - Team Splice")
        
        # Recreate ambulance view as underlay FIRST
        # try:
        #     print("Recreating ambulance view")
        #     self.ambulance_overlay = AmbulanceOverlay(self.root, w, h)
        #     self.ambulance_overlay.place(0, 0)  # Place at the very bottom layer
        #     print("Ambulance view recreated and placed")
        # except Exception as e:
        #     print("Exception recreating ambulance view:", e)
        
        # Recreate the menu bar AFTER overlay so it appears on top
        self.create_menu_bar()
        
        # Recreate button menu
        try:
            print("Recreating button menu")
            self.button_menu = ButtonMenu(self.root, self.user_buttons)
            print("Button menu recreated")
        except Exception as e:
            print("Exception recreating button menu:", e)
        
        # Recreate left/right button menus
        try:
            print("Recreating left/right button menus")
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
            print("Left/right button menus recreated")
        except Exception as e:
            print("Exception recreating left/right button menus:", e)
        
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
        init_h = 8
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
        
        movement_font = font.Font(family="Fixedsys", size=10)
        # Recreate movement label
        self.movement_label = tk.Label(self.root, text=f"Route Progress: {self.movement_count}/20", 
                                      font=movement_font, bg="#5B7B7A", fg="#FFFFFF",
                                      relief="groove", bd=2, padx=10, pady=5)
        self.movement_label.place(x=1060, y=460)
        
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
        
        # Recreate CNN labels
        try:
            # Get imperfect predictions for display
            left_image_path = os.path.join(self.data_fp, self.image_left.Filename)
            right_image_path = os.path.join(self.data_fp, self.image_right.Filename)
            
            left_text = self.imperfect_cnn.get_display_text(left_image_path, "LEFT")
            right_text = self.imperfect_cnn.get_display_text(right_image_path, "RIGHT")
            
            # Create text labels above images
            from ui_elements.theme import LABEL_FONT
            self.left_cnn_label = tk.Label(self.root, text=left_text, font=LABEL_FONT, 
                                         bg="black", fg="yellow", wraplength=image_width-10)
            self.right_cnn_label = tk.Label(self.root, text=right_text, font=LABEL_FONT, 
                                          bg="black", fg="yellow", wraplength=image_width-10)
            
            # Place labels above images
            self.left_cnn_label.place(x=center_x - offset, y=y_top - 40)
            self.right_cnn_label.place(x=center_x - offset + image_width + horizontal_gap, y=y_top - 40)
            print("CNN labels recreated in restore_main_ui")
        except Exception as e:
            print("Exception recreating CNN labels in restore_main_ui:", e)
        
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
        
        # Import theme fonts
        from ui_elements.theme import UPGRADE_FONT, LABEL_FONT
        
        # Title label
        if action_name == "Save":
            title_label = tk.Label(main_frame, text=f"Saving", 
                              font=UPGRADE_FONT, fg="#2E86AB")
            title_label.pack(pady=(0, 15))
        else:
            title_label = tk.Label(main_frame, text=f"{action_name}ing", 
                              font=UPGRADE_FONT, fg="#2E86AB")
            title_label.pack(pady=(0, 15))
        
        # Instructions text with highlighting
        instructions_label = tk.Label(main_frame, text=instructions_text, 
                                    justify="center", anchor="center", font=LABEL_FONT,
                                    fg="#E74C3C", bg="#FDF2E9", 
                                    relief="solid", bd=1, padx=15, pady=15)
        instructions_label.pack(expand=True, fill="both", pady=(0, 15))
        
        # Close button
        close_btn = tk.Button(main_frame, text="Close", 
                             command=popup.destroy,
                             font=LABEL_FONT, bg="#3498DB", fg="white",
                             relief="raised", bd=2, padx=20, pady=5)
        close_btn.pack()


    def show_rules(self):
        self.clear_main_ui()
        # Set the root background to black
        self.root.configure(bg="black")
        
        # Create a canvas to hold the background image and UI elements
        canvas = tk.Canvas(self.root, width=1280, height=800, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        
        # Load and resize the city background image
        try:
            image_path = "images/city_background.png"
            original_image = Image.open(image_path)
            
            # Resize image to fit the window (1280x800)
            resized_image = original_image.resize((1280, 800), Image.Resampling.LANCZOS)
            self.bg_image_rules = ImageTk.PhotoImage(resized_image)  # Store reference
            
            # Place the background image on the canvas
            canvas.create_image(0, 0, image=self.bg_image_rules, anchor="nw")
            
            # Add a semi-transparent overlay to make text more readable
            canvas.create_rectangle(0, 0, 1280, 800, fill="black", stipple="gray50")
            
        except Exception as e:
            print(f"Could not load city background image: {e}")
            # Fallback to a solid color background
            canvas.configure(bg="darkblue")
        
        # Title with start screen styling
        canvas.create_text(640, 60, text="Game Rules", 
                          font=("Arial", 18, "bold"), 
                          fill="white", anchor="center")
        
        rules_text = (
            "Welcome to Team Splice (SGAI 2025)!\n\n"
            
            "🎯 OBJECTIVE:\n"
            "Complete the ambulance route (20 movements) while saving humans and eliminating zombies.\n\n"
            
            "🎮 GAMEPLAY:\n"
            "• You'll encounter pairs of humanoids (humans and zombies)\n"
            "• Choose your actions carefully - you can only save one set per scenario\n"
            "• Use the (risky) advice at the top of your screen to help you make decisions\n"
            "• Complete your route before time runs out\n\n"
            
            "🔧 ACTIONS:\n"
            "• SKIP: Pass on the current humanoids\n"
            "• INSPECT: Get more information (costs time)\n"
            "• SQUISH: Eliminate zombies (costs time)\n"
            "• SAVE: Rescue humans (costs time and capacity)\n"
            "• SCRAM: Return to base when capacity is full\n\n"
            
            "📊 SCORING:\n"
            "• +Points: Saving humans, eliminating zombies\n"
            "• -Points: Saving zombies, eliminating humans\n"
            "• Money earned from saving humans can be used for upgrades\n\n"
            
            "⏰ TIME MANAGEMENT:\n"
            "• You have 12 hours (720 minutes) to complete your route\n"
            "• Route progress is shown in the center of your screen\n"
            "• If you don't complete the route on time, you lose points\n\n"
            
            "💡 TIPS:\n"
            "• Use INSPECT to gather information before making decisions\n"
            "• Manage your capacity wisely - you can't save everyone\n"
            "• Watch the clock and plan your route completion\n"
            "• Different classes have different abilities: use them wisely!\n"
            "• Remember: You can only save one set per scenario!"
        )

        # Create a scrollable text widget for rules
        text_frame = tk.Frame(canvas, bg="#1a1a1a")
        canvas.create_window(640, 400, window=text_frame, anchor="center", width=1000, height=500)
        
        # Create text widget with scrollbar
        text_widget = tk.Text(text_frame, wrap="word", font=("Arial", 13), 
                             bg="#1a1a1a", fg="white", relief="flat",
                             padx=30, pady=30, state="normal")
        scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Pack the text widget and scrollbar
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Insert the rules text
        text_widget.insert("1.0", rules_text)
        text_widget.config(state="disabled")  # Make read-only

        # Back button with start screen styling
        back_button = tk.Button(canvas, text="Back to Game", 
                               font=("Arial", 14, "bold"), 
                               fg="white", bg="#5B7B7A",  # Green background like start screen
                               relief="raised", bd=3, width=15, height=1,
                               command=self.restore_main_ui)
        canvas.create_window(640, 700, window=back_button, anchor="center")

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

        # Removed time warning popup - no longer showing warning dialog

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
            
            # Import fonts needed for both cases
            from ui_elements.theme import FINAL_FONT_HEADER, FINAL_FONT_SUB
            
            # 'Game Complete' label 
            if route_complete:
                final_score = self.scorekeeper.get_final_score(route_complete=True)
                game_complete_label = tk.Label(self.final_score_frame, text="Route Complete", font=FINAL_FONT_HEADER, fg="white", bg="black", highlightthickness=0)
                game_complete_label.pack(pady=(10, 5))
                final_score_label = tk.Label(self.final_score_frame, text="FINAL SCORE: " + f" {final_score}", font=FINAL_FONT_SUB, fg="white", bg="black", highlightthickness=0)
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
                game_complete_label = tk.Label(self.final_score_frame, text="Game Complete", font=FINAL_FONT_HEADER, fg="white", bg="black", highlightthickness=0)
                game_complete_label.pack(pady=(10, 5))
                final_score_label = tk.Label(self.final_score_frame, text="FINAL SCORE: " + f" {final_score}", font=FINAL_FONT_SUB, fg="white", bg="black", highlightthickness=0)
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
            from ui_elements.theme import LABEL_FONT
            killed_label = tk.Label(self.final_score_frame, text=f"Humans Killed: {scorekeeper.get_score(self.image_left, self.image_right)['killed']}", font=LABEL_FONT, fg="white", bg="black", highlightthickness=0)
            killed_label.pack()
            saved_label = tk.Label(self.final_score_frame, text=f"Humans Saved: {scorekeeper.get_score(self.image_left, self.image_right)['saved']}", font=LABEL_FONT, fg="white", bg="black", highlightthickness=0)
            saved_label.pack()
            
            zombie_ambu = tk.Label(self.final_score_frame, text=f"Zombies in Ambulance (Saved): {self.scorekeeper.false_saves}", font=LABEL_FONT, fg="white", bg="black", highlightthickness=0)
            zombie_ambu.pack()

            # Add zombies killed display
            zombies_killed_label = tk.Label(self.final_score_frame, text=f"Zombies Killed: {self.scorekeeper.scorekeeper['zombie_killed']}", font=LABEL_FONT, fg="white", bg="black", highlightthickness=0)
            zombies_killed_label.pack()

            # accuracy_label = tk.Label(self.final_score_frame, text=f"Accuracy: {accuracy:.2f}%", font=("Arial", 12), fg="white", bg="black", highlightthickness=0)
            # accuracy_label.pack()
            zombies_saved_score_label = tk.Label(self.final_score_frame,
                text=f"Zombies Saved: {self.scorekeeper.false_saves}",
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
        
        # Reset UI elements to reflect the reset upgrade values
        # 1. Reset capacity meter to show 10 slots (base capacity)
        if hasattr(self, 'capacity_meter'):
            self.capacity_meter.reset_capacity(10)
        
        # 2. Reset button texts to reflect original times
        # Scram time: 5 mins per route slot (base time)
        # Inspect time: 15 mins (base time)
        self.update_button_texts()
        
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
        
        # Recreate CNN labels for new images
        try:
            if hasattr(self, 'left_cnn_label') and hasattr(self, 'right_cnn_label'):
                # Update existing CNN labels with new images
                left_text = self.imperfect_cnn.get_display_text(fp_left, "LEFT")
                right_text = self.imperfect_cnn.get_display_text(fp_right, "RIGHT")
                
                self.left_cnn_label.config(text=left_text)
                self.right_cnn_label.config(text=right_text)
                print("CNN labels updated in reset_game")
            else:
                print("CNN labels not found during reset_game, may need recreation")
        except Exception as e:
            print("Exception updating CNN labels in reset_game:", e)
        
        # 5. Reset scram cost to starting value
        if hasattr(self.scorekeeper, 'scram_time_reduction'):
            self.scorekeeper.scram_time_reduction = 0
        self.button_menu.buttons[4].config(text="Scram (5 mins)")
        # 5. Update button texts to reflect reset upgrade values
        # The upgrade manager reset should have already handled this, but let's make sure
        self.update_button_texts()
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
        bg_img = ImageTk.PhotoImage(Image.open('images/ChatgptMap.png').resize((canvas_width, canvas_height)))
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
            self.map_canvas.create_oval(px-3, py-3, px+3, py+3, fill="#3498db", outline="")
        # Draw ambulance at current position
        r, c = self.ambulance_pos
        x = c * self.cell_size + 10 + self.cell_size // 2
        y = r * self.cell_size + 10 + self.cell_size // 2
        self.map_canvas.create_oval(x-14, y-14, x+14, y+14, fill="#fff", outline="#3498db", width=3)
        from ui_elements.theme import LABEL_FONT
        self.map_canvas.create_text(x, y, text="🚑", font=LABEL_FONT)

    def move_ambulance_by_cell(self):
        # Don't move if route is already complete
        if self.route_complete:
            return
            
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
            
        # Only increment movement count if the move is valid
        if 0 <= new_r < self.grid_rows and 0 <= new_c < self.grid_cols and self.map_array[new_r][new_c] != 0:
            self.ambulance_pos = [new_r, new_c]
            self.path_history.append(tuple(self.ambulance_pos))
            # Increment movement counter
            self.movement_count += 1
            # print(f"[DEBUG] Ambulance movement {self.movement_count}/20")
            
            # Update movement progress label
            self.movement_label.config(text=f"Route Progress: {self.movement_count}/20")
            
            # Check if we've reached exactly 20 movements
            if self.movement_count == 20:
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
        # Import theme fonts
        from ui_elements.theme import FINAL_FONT_HEADER, FINAL_FONT_SUB, LABEL_FONT, UPGRADE_FONT
        
        # 'Route Complete' label
        game_complete_label = tk.Label(self.final_score_frame, text="Route Complete", font=FINAL_FONT_HEADER,fg="white",bg="black",highlightthickness=0)
        game_complete_label.pack(pady=(10, 5))
        # 'Final Score' label
        final_score_label = tk.Label(self.final_score_frame, text=f"FINAL SCORE: {final_score}", font=FINAL_FONT_SUB,fg="white",bg="black",highlightthickness=0)
        final_score_label.pack(pady=(5, 2))
        # Scoring details
        killed_label = tk.Label(self.final_score_frame, text=f"Humans Killed: {self.scorekeeper.get_score(self.image_left, self.image_right)['killed']}", font=LABEL_FONT,fg="white",bg="black",highlightthickness=0)
        killed_label.pack()
        saved_label = tk.Label(self.final_score_frame, text=f"Humans Saved: {self.scorekeeper.get_score(self.image_left, self.image_right)['saved']}", font=LABEL_FONT,fg="white",bg="black",highlightthickness=0)
        saved_label.pack()
        zombie_ambu = tk.Label(self.final_score_frame, text=f"Zombies in Ambulance: {self.scorekeeper.false_saves}", font=LABEL_FONT, fg="white", bg="black", highlightthickness=0)
        zombie_ambu.pack()

        zombies_saved_score_label = tk.Label(self.final_score_frame,
            text=f"Zombies in Ambulance (Saved): {self.scorekeeper.false_saves}",
            font=LABEL_FONT,
            fg="white",
            bg="black",
            highlightthickness=0)
        zombies_saved_score_label.pack()

        # Add zombies killed display
        zombies_killed_label = tk.Label(self.final_score_frame, text=f"Zombies Killed: {self.scorekeeper.scorekeeper['zombie_killed']}", font=LABEL_FONT, fg="white", bg="black", highlightthickness=0)
        zombies_killed_label.pack()

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
        self.replay_btn = tk.Button(self.final_score_frame, text="Replay", font=LABEL_FONT, command=lambda: self.reset_game(self.data_parser, self.data_fp))
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
        
        # Create a canvas to hold the background image and UI elements
        canvas = tk.Canvas(self.root, width=1280, height=800, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        
        # Load and resize the city background image
        try:
            image_path = "images/city_background.png"
            original_image = Image.open(image_path)
            
            # Resize image to fit the window (1280x800)
            resized_image = original_image.resize((1280, 800), Image.Resampling.LANCZOS)
            self.bg_image_upgrades = ImageTk.PhotoImage(resized_image)  # Store reference
            
            # Place the background image on the canvas
            canvas.create_image(0, 0, image=self.bg_image_upgrades, anchor="nw")
            
            # Add a semi-transparent overlay to make text more readable
            canvas.create_rectangle(0, 0, 1280, 800, fill="black", stipple="gray50")
            
        except Exception as e:
            print(f"Could not load city background image: {e}")
            # Fallback to a solid color background
            canvas.configure(bg="darkblue")

        # Title with start screen styling
        from ui_elements.theme import TITLE_FONT_FOR_CURRENT_TIME, UPGRADE_FONT
        canvas.create_text(640, 60, text="Upgrade Shop", 
                          font=TITLE_FONT_FOR_CURRENT_TIME, 
                          fill="white", anchor="center")

        # Money display with start screen styling
        money = self.scorekeeper.upgrade_manager.get_money()
        money_label = tk.Label(canvas, text=f"Money: ${money}", 
                              font=UPGRADE_FONT, 
                              fg="white", bg="#1a1a1a")
        canvas.create_window(640, 120, window=money_label, anchor="center")

        # Create a frame for upgrades with dark background
        upgrades_frame = tk.Frame(canvas, bg="#1a1a1a")
        canvas.create_window(640, 400, window=upgrades_frame, anchor="center")

        # Each upgrade
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
                btn = tk.Button(upgrades_frame, text=btn_text, 
                               font=UPGRADE_FONT,
                               state='disabled', bg="#4A6B6A", fg="gray",
                               relief="raised", bd=3, width=50, height=2)
            else:
                btn_text += f" - ${cost}"
                btn = tk.Button(upgrades_frame, text=btn_text, 
                               font=UPGRADE_FONT,
                               command=make_purchase, 
                               fg="white", bg="#5B7B7A",  # Green background like start screen
                               relief="raised", bd=3, width=50, height=2)

            btn.pack(pady=5)

        # Back button with start screen styling
        back_button = tk.Button(canvas, text="Back to Game", 
                               font=UPGRADE_FONT, 
                               fg="white", bg="#5B7B7A",  # Green background like start screen
                               relief="raised", bd=3, width=15, height=1,
                               command=self.restore_main_ui)
        canvas.create_window(640, 700, window=back_button, anchor="center")

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
        from ui_elements.theme import LABEL_FONT
        canvas.create_text(10, 10, anchor='nw', text=text, font=LABEL_FONT)

    def add_elapsed_time(self, minutes):
        self.scorekeeper.remaining_time -= minutes

    def create_menu_bar(self):
        self.menu_bar = tk.Frame(self.root, bg="#000000", height=50)
        self.menu_bar.place(x=0, y=0, width=1280, height=50)

    # Styled buttons
        from ui_elements.theme import UPGRADE_FONT, RULES_FONT

        self.upgrade_btn = tk.Button(self.menu_bar, text="Upgrades", font=UPGRADE_FONT, bg="#ffffff", fg="#2E86AB",
              relief="raised", padx=10, pady=5, command=self.show_upgrade_shop)
        self.upgrade_btn.pack(side="left", padx=1)

        self.rules_btn = tk.Button(self.menu_bar, text="Rules", font=RULES_FONT, bg="#ffffff", fg="#2E86AB",
              relief="raised", padx=10, pady=5, command=self.show_rules)
        self.rules_btn.pack(side="left", padx=1)

        tk.Button(self.menu_bar, text="Exit", font=UPGRADE_FONT, bg="#ffffff", fg="#AA0000",
              relief="raised", padx=10, pady=5, command=self.root.quit).pack(side="left", padx=1)
