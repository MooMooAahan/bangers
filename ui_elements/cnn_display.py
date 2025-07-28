"""
User-Facing CNN Display
Shows imperfect CNN predictions above images in the game UI
Configurable accuracy for educational purposes
"""

import tkinter as tk
from tkinter import ttk
import random

from endpoints.enhanced_predictor import get_enhanced_predictor, ImperfectCNNDisplay
from gameplay.ai_decision_logger import get_ai_logger


class CNNDisplayWidget:
    """
    Widget that displays CNN predictions above game images
    Shows intentionally imperfect predictions to simulate real-world AI uncertainty
    """
    
    def __init__(self, parent_frame, accuracy_rate=0.5):
        self.parent_frame = parent_frame
        self.accuracy_rate = accuracy_rate
        
        # Get predictor and display handler
        self.enhanced_predictor = get_enhanced_predictor()
        self.imperfect_display = ImperfectCNNDisplay(self.enhanced_predictor, accuracy_rate)
        self.ai_logger = get_ai_logger()
        
        # Create UI elements
        self.setup_ui()
        
        # Current predictions
        self.current_left_prediction = None
        self.current_right_prediction = None
        
    def setup_ui(self):
        """Setup the CNN display UI elements"""
        
        # Main frame for CNN display
        self.cnn_frame = tk.Frame(self.parent_frame, bg='lightgray', relief='raised', bd=2)
        self.cnn_frame.pack(fill='x', padx=5, pady=5)
        
        # Title
        title_label = tk.Label(self.cnn_frame, text="🤖 AI Assistant (Imperfect)", 
                              font=('Arial', 12, 'bold'), bg='lightgray')
        title_label.pack(pady=5)
        
        # Accuracy control
        accuracy_frame = tk.Frame(self.cnn_frame, bg='lightgray')
        accuracy_frame.pack(fill='x', padx=10, pady=2)
        
        tk.Label(accuracy_frame, text="AI Accuracy:", bg='lightgray', font=('Arial', 10)).pack(side='left')
        
        self.accuracy_var = tk.DoubleVar(value=self.accuracy_rate)
        self.accuracy_scale = tk.Scale(accuracy_frame, from_=0.3, to=0.7, resolution=0.1,
                                      orient='horizontal', variable=self.accuracy_var,
                                      command=self.on_accuracy_changed, bg='lightgray')
        self.accuracy_scale.pack(side='left', padx=10)
        
        self.accuracy_label = tk.Label(accuracy_frame, text=f"{int(self.accuracy_rate*100)}%", 
                                      bg='lightgray', font=('Arial', 10, 'bold'))
        self.accuracy_label.pack(side='left')
        
        # Predictions display frame
        self.predictions_frame = tk.Frame(self.cnn_frame, bg='lightgray')
        self.predictions_frame.pack(fill='x', padx=10, pady=5)
        
        # Left side prediction
        self.left_frame = tk.Frame(self.predictions_frame, bg='lightblue', relief='solid', bd=1)
        self.left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        tk.Label(self.left_frame, text="⬅️ Left Side", font=('Arial', 10, 'bold'), 
                bg='lightblue').pack(pady=2)
        
        self.left_status_label = tk.Label(self.left_frame, text="Status: Unknown", 
                                         font=('Arial', 9), bg='lightblue')
        self.left_status_label.pack()
        
        self.left_occupation_label = tk.Label(self.left_frame, text="Role: Unknown", 
                                             font=('Arial', 9), bg='lightblue')
        self.left_occupation_label.pack()
        
        self.left_confidence_label = tk.Label(self.left_frame, text="Confidence: --", 
                                             font=('Arial', 8), bg='lightblue', fg='gray')
        self.left_confidence_label.pack()
        
        # Right side prediction
        self.right_frame = tk.Frame(self.predictions_frame, bg='lightcoral', relief='solid', bd=1)
        self.right_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        tk.Label(self.right_frame, text="➡️ Right Side", font=('Arial', 10, 'bold'), 
                bg='lightcoral').pack(pady=2)
        
        self.right_status_label = tk.Label(self.right_frame, text="Status: Unknown", 
                                          font=('Arial', 9), bg='lightcoral')
        self.right_status_label.pack()
        
        self.right_occupation_label = tk.Label(self.right_frame, text="Role: Unknown", 
                                              font=('Arial', 9), bg='lightcoral')
        self.right_occupation_label.pack()
        
        self.right_confidence_label = tk.Label(self.right_frame, text="Confidence: --", 
                                              font=('Arial', 8), bg='lightcoral', fg='gray')
        self.right_confidence_label.pack()
        
        # Warning label
        warning_label = tk.Label(self.cnn_frame, 
                                text="⚠️ AI predictions may be incorrect - Use 'Inspect' for accurate info",
                                font=('Arial', 8), bg='lightgray', fg='red')
        warning_label.pack(pady=2)
        
    def on_accuracy_changed(self, value):
        """Handle accuracy slider change"""
        self.accuracy_rate = float(value)
        self.accuracy_label.config(text=f"{int(self.accuracy_rate*100)}%")
        self.imperfect_display.set_accuracy(self.accuracy_rate)
        
        # Update current predictions with new accuracy
        if hasattr(self, 'current_left_image') and hasattr(self, 'current_right_image'):
            self.update_predictions(self.current_left_image, self.current_right_image)
    
    def update_predictions(self, left_image_path, right_image_path):
        """Update CNN predictions for new images"""
        
        self.current_left_image = left_image_path
        self.current_right_image = right_image_path
        
        try:
            # Get true predictions first
            true_left = self.enhanced_predictor.predict_combined(left_image_path)
            true_right = self.enhanced_predictor.predict_combined(right_image_path)
            
            # Get imperfect predictions for display
            left_prediction = self.enhanced_predictor.get_imperfect_prediction(
                left_image_path, self.accuracy_rate)
            right_prediction = self.enhanced_predictor.get_imperfect_prediction(
                right_image_path, self.accuracy_rate)
            
            self.current_left_prediction = left_prediction
            self.current_right_prediction = right_prediction
            
            # Update UI
            self._update_left_display(left_prediction)
            self._update_right_display(right_prediction)
            
            # Log the imperfect display
            self.ai_logger.log_imperfect_cnn_display(
                left_image_path, 'left', true_left, left_prediction, self.accuracy_rate)
            self.ai_logger.log_imperfect_cnn_display(
                right_image_path, 'right', true_right, right_prediction, self.accuracy_rate)
                
        except Exception as e:
            print(f"Error updating CNN predictions: {e}")
            self._show_error_display()
    
    def _update_left_display(self, prediction):
        """Update left side prediction display"""
        
        status = prediction['status'].title()
        occupation = prediction['occupation']
        is_correct = prediction.get('is_correct', True)
        
        # Color coding based on correctness (for debugging, not shown to user)
        status_color = 'black' if is_correct else 'darkred'
        
        self.left_status_label.config(text=f"Status: {status}", fg=status_color)
        self.left_occupation_label.config(text=f"Role: {occupation}", fg=status_color)
        
        # Fake confidence (for realism)
        confidence = random.randint(65, 95)
        self.left_confidence_label.config(text=f"Confidence: {confidence}%")
    
    def _update_right_display(self, prediction):
        """Update right side prediction display"""
        
        status = prediction['status'].title()
        occupation = prediction['occupation']
        is_correct = prediction.get('is_correct', True)
        
        # Color coding based on correctness (for debugging, not shown to user)
        status_color = 'black' if is_correct else 'darkred'
        
        self.right_status_label.config(text=f"Status: {status}", fg=status_color)
        self.right_occupation_label.config(text=f"Role: {occupation}", fg=status_color)
        
        # Fake confidence (for realism)
        confidence = random.randint(65, 95)
        self.right_confidence_label.config(text=f"Confidence: {confidence}%")
    
    def _show_error_display(self):
        """Show error state in display"""
        
        self.left_status_label.config(text="Status: Error", fg='red')
        self.left_occupation_label.config(text="Role: Error", fg='red')
        self.left_confidence_label.config(text="Confidence: --")
        
        self.right_status_label.config(text="Status: Error", fg='red')
        self.right_occupation_label.config(text="Role: Error", fg='red')
        self.right_confidence_label.config(text="Confidence: --")
    
    def clear_predictions(self):
        """Clear all predictions"""
        
        self.left_status_label.config(text="Status: Unknown", fg='black')
        self.left_occupation_label.config(text="Role: Unknown", fg='black')
        self.left_confidence_label.config(text="Confidence: --")
        
        self.right_status_label.config(text="Status: Unknown", fg='black')
        self.right_occupation_label.config(text="Role: Unknown", fg='black')
        self.right_confidence_label.config(text="Confidence: --")
        
        self.current_left_prediction = None
        self.current_right_prediction = None
    
    def get_current_predictions(self):
        """Get current predictions for logging/analysis"""
        return {
            'left': self.current_left_prediction,
            'right': self.current_right_prediction,
            'accuracy_setting': self.accuracy_rate
        }
    
    def hide(self):
        """Hide the CNN display"""
        self.cnn_frame.pack_forget()
    
    def show(self):
        """Show the CNN display"""
        self.cnn_frame.pack(fill='x', padx=5, pady=5)
