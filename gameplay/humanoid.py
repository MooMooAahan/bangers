from gameplay.enums import State

MAP_CLASS_STR_TO_INT = {s.value:i for i,s in enumerate(State)}
MAP_CLASS_INT_TO_STR = [s.value for s in State]

# Enhanced 21-class system: status_occupation combinations
ENHANCED_CLASSES = [
    # Zombie variants
    'zombie_civilian', 'zombie_child', 'zombie_doctor', 'zombie_police', 'zombie_militant',
    # Healthy variants  
    'healthy_civilian', 'healthy_child', 'healthy_doctor', 'healthy_police', 'healthy_militant',
    # Injured variants
    'injured_civilian', 'injured_child', 'injured_doctor', 'injured_police', 'injured_militant',
    # Corpse variants
    'corpse_civilian', 'corpse_child', 'corpse_doctor', 'corpse_police', 'corpse_militant',
    # No person
    'no_person'
]

ENHANCED_CLASS_TO_INT = {cls: i for i, cls in enumerate(ENHANCED_CLASSES)}
ENHANCED_INT_TO_CLASS = {i: cls for i, cls in enumerate(ENHANCED_CLASSES)}

class Humanoid(object):
    """
    Are they a human or a zombie???
    Enhanced with status + occupation classification
    """
    
    def __init__(self, fp, state, role, value = 0):
        self.fp = fp
        self.state = state
        self.role = role
        # self.value = value

    def is_zombie(self):
        return self.state == State.ZOMBIE.value

    def is_injured(self):
        return self.state == State.INJURED.value

    def is_healthy(self):
        return self.state == State.HEALTHY.value

    def is_corpse(self):
        return self.state == State.CORPSE.value

    def get_role(self):
        return self.role
    
    @staticmethod
    def get_state_idx(class_string):
        return MAP_CLASS_STR_TO_INT[class_string]
    
    @staticmethod
    def get_state_string(class_idx):
        return MAP_CLASS_INT_TO_STR[class_idx]
    
    @staticmethod
    def get_all_states():
        return [s.value for s in State]
    
    # Enhanced class system methods
    @staticmethod
    def get_enhanced_classes():
        """Returns all 21 enhanced classes"""
        return ENHANCED_CLASSES
    
    @staticmethod
    def get_enhanced_class_count():
        """Returns number of enhanced classes (21)"""
        return len(ENHANCED_CLASSES)
    
    @staticmethod
    def create_enhanced_class(status, occupation):
        """
        Create enhanced class from status and occupation
        Args:
            status: 'zombie', 'healthy', 'injured', 'corpse'
            occupation: 'civilian', 'child', 'doctor', 'police', 'militant'
        Returns:
            Combined class like 'zombie_police'
        """
        # Handle edge cases
        if occupation.lower() in ['blank', 'unknown', '']:
            occupation = 'civilian'  # Default occupation
            
        enhanced_class = f"{status.lower()}_{occupation.lower()}"
        
        # Validate class exists
        if enhanced_class not in ENHANCED_CLASSES:
            print(f"Warning: Unknown enhanced class {enhanced_class}, defaulting to healthy_civilian")
            return 'healthy_civilian'
            
        return enhanced_class
    
    @staticmethod
    def parse_enhanced_class(enhanced_class):
        """
        Parse enhanced class back to status and occupation
        Args:
            enhanced_class: 'zombie_police'
        Returns:
            tuple: ('zombie', 'police')
        """
        if enhanced_class == 'no_person':
            return ('no_person', 'none')
            
        if '_' not in enhanced_class:
            return ('healthy', 'civilian')  # Default fallback
            
        parts = enhanced_class.split('_', 1)
        return (parts[0], parts[1])
    
    @staticmethod
    def enhanced_class_to_int(enhanced_class):
        """Convert enhanced class to integer index"""
        return ENHANCED_CLASS_TO_INT.get(enhanced_class, 5)  # Default to healthy_civilian
    
    @staticmethod
    def int_to_enhanced_class(class_idx):
        """Convert integer index to enhanced class"""
        return ENHANCED_INT_TO_CLASS.get(class_idx, 'healthy_civilian')

