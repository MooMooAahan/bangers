from gameplay.enums import State

MAP_CLASS_STR_TO_INT = {s.value:i for i,s in enumerate(State)}
MAP_CLASS_INT_TO_STR = [s.value for s in State]

class Humanoid(object):
    """
    Are they a human or a zombie???
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
        return MAP_CLASS_INT_TO_STR

