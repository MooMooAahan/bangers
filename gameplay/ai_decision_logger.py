"""
AI Decision Logger
Extends the existing logging system to track AI (RL/CNN) decisions
Modular design for easy extension and analysis
"""

import pandas as pd
import os
from datetime import datetime
import json


class AIDecisionLogger:
    """
    Logger for AI decisions (RL agent and CNN predictions)
    Extends the existing human decision logging system
    """
    
    def __init__(self, log_file_prefix="ai_decisions"):
        self.log_file_prefix = log_file_prefix
        self.ai_decisions = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_rl_decision(self, image_left_path, image_right_path, 
                       cnn_predictions, rl_action, rl_confidence, 
                       game_state, reward_info=None):
        """
        Log RL agent decision
        
        Args:
            image_left_path: Path to left image
            image_right_path: Path to right image
            cnn_predictions: Dict with CNN predictions for both sides
            rl_action: Action taken by RL agent
            rl_confidence: Confidence scores for all actions
            game_state: Current game state (time, capacity, etc.)
            reward_info: Reward calculation details
        """
        
        decision_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'decision_type': 'RL_AGENT',
            'image_left': image_left_path,
            'image_right': image_right_path,
            
            # CNN predictions
            'cnn_left_status': cnn_predictions.get('left', {}).get('status', 'unknown'),
            'cnn_left_occupation': cnn_predictions.get('left', {}).get('occupation', 'unknown'),
            'cnn_right_status': cnn_predictions.get('right', {}).get('status', 'unknown'),
            'cnn_right_occupation': cnn_predictions.get('right', {}).get('occupation', 'unknown'),
            
            # RL decision
            'rl_action': rl_action,
            'rl_action_name': self._action_idx_to_name(rl_action),
            'rl_confidence': json.dumps(rl_confidence.tolist() if hasattr(rl_confidence, 'tolist') else rl_confidence),
            
            # Game state
            'remaining_time': game_state.get('remaining_time', 0),
            'ambulance_capacity_used': game_state.get('capacity_used', 0),
            'ambulance_zombie_count': game_state.get('zombie_count', 0),
            'ambulance_healthy_count': game_state.get('healthy_count', 0),
            'ambulance_injured_count': game_state.get('injured_count', 0),
            
            # Reward information
            'immediate_reward': reward_info.get('immediate_reward', 0) if reward_info else 0,
            'cumulative_reward': reward_info.get('cumulative_reward', 0) if reward_info else 0,
        }
        
        self.ai_decisions.append(decision_entry)
        
    def log_cnn_prediction(self, image_path, side, true_labels, 
                          cnn_predictions, model_type="status"):
        """
        Log CNN prediction for analysis
        
        Args:
            image_path: Path to image
            side: 'left' or 'right'
            true_labels: Ground truth labels
            cnn_predictions: CNN prediction results
            model_type: 'status' or 'occupation'
        """
        
        prediction_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'decision_type': f'CNN_{model_type.upper()}',
            'image_path': image_path,
            'side': side,
            
            # Ground truth
            'true_status': true_labels.get('status', 'unknown'),
            'true_occupation': true_labels.get('occupation', 'unknown'),
            
            # CNN predictions
            'predicted_status': cnn_predictions.get('status', 'unknown'),
            'predicted_occupation': cnn_predictions.get('occupation', 'unknown'),
            'prediction_confidence': json.dumps(cnn_predictions.get('probabilities', [])),
            
            # Accuracy
            'status_correct': true_labels.get('status') == cnn_predictions.get('status'),
            'occupation_correct': true_labels.get('occupation') == cnn_predictions.get('occupation'),
        }
        
        self.ai_decisions.append(prediction_entry)
        
    def log_imperfect_cnn_display(self, image_path, side, true_prediction, 
                                 displayed_prediction, accuracy_setting):
        """
        Log imperfect CNN display for user interface
        
        Args:
            image_path: Path to image
            side: 'left' or 'right'
            true_prediction: Actual CNN prediction
            displayed_prediction: What was shown to user (may be wrong)
            accuracy_setting: Current accuracy setting (0.4-0.6)
        """
        
        display_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'decision_type': 'IMPERFECT_CNN_DISPLAY',
            'image_path': image_path,
            'side': side,
            
            # True vs displayed
            'true_status': true_prediction.get('status', 'unknown'),
            'true_occupation': true_prediction.get('occupation', 'unknown'),
            'displayed_status': displayed_prediction.get('status', 'unknown'),
            'displayed_occupation': displayed_prediction.get('occupation', 'unknown'),
            
            # Settings and accuracy
            'accuracy_setting': accuracy_setting,
            'was_correct': displayed_prediction.get('is_correct', True),
            'status_match': true_prediction.get('status') == displayed_prediction.get('status'),
            'occupation_match': true_prediction.get('occupation') == displayed_prediction.get('occupation'),
        }
        
        self.ai_decisions.append(display_entry)
        
    def save_log(self, final=False):
        """
        Save AI decisions to CSV file
        
        Args:
            final: Whether this is the final save (end of session)
        """
        
        if not self.ai_decisions:
            return
            
        # Create DataFrame
        df = pd.DataFrame(self.ai_decisions)
        
        # Generate filename
        suffix = "_final" if final else ""
        filename = f"{self.log_file_prefix}_{self.session_id}{suffix}.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"💾 AI decisions saved to {filename} ({len(self.ai_decisions)} entries)")
        
        # Also append to master log if it exists
        master_log = "ai_decisions_master.csv"
        if os.path.exists(master_log):
            df.to_csv(master_log, mode='a', header=False, index=False)
        else:
            df.to_csv(master_log, index=False)
            
    def get_analysis_summary(self):
        """
        Get summary statistics for analysis
        """
        
        if not self.ai_decisions:
            return "No AI decisions logged yet."
            
        df = pd.DataFrame(self.ai_decisions)
        
        summary = {
            'total_decisions': len(df),
            'rl_decisions': len(df[df['decision_type'] == 'RL_AGENT']),
            'cnn_predictions': len(df[df['decision_type'].str.contains('CNN')]),
            'imperfect_displays': len(df[df['decision_type'] == 'IMPERFECT_CNN_DISPLAY']),
        }
        
        # RL action distribution
        if 'rl_action_name' in df.columns:
            rl_actions = df[df['decision_type'] == 'RL_AGENT']['rl_action_name'].value_counts()
            summary['rl_action_distribution'] = rl_actions.to_dict()
            
        # CNN accuracy
        cnn_df = df[df['decision_type'].str.contains('CNN', na=False)]
        if len(cnn_df) > 0:
            if 'status_correct' in cnn_df.columns:
                summary['cnn_status_accuracy'] = cnn_df['status_correct'].mean()
            if 'occupation_correct' in cnn_df.columns:
                summary['cnn_occupation_accuracy'] = cnn_df['occupation_correct'].mean()
                
        return summary
        
    def _action_idx_to_name(self, action_idx):
        """Convert action index to readable name"""
        action_names = {
            0: "SKIP_BOTH",
            1: "SQUISH_LEFT", 
            2: "SQUISH_RIGHT",
            3: "SAVE_LEFT",
            4: "SAVE_RIGHT",
            5: "SCRAM"
        }
        return action_names.get(action_idx, f"UNKNOWN_{action_idx}")
        
    def reset_session(self):
        """Start a new logging session"""
        self.save_log(final=True)
        self.ai_decisions = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")


# Global instance for easy access
_ai_logger = None

def get_ai_logger():
    """Get global AI decision logger instance"""
    global _ai_logger
    if _ai_logger is None:
        _ai_logger = AIDecisionLogger()
    return _ai_logger
