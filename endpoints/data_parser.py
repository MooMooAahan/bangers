import random
import pandas as pd
from gameplay.humanoid import Humanoid
from gameplay.enums import State
import os


class DataParser(object):
    """
    Parses the input data photos and assigns their file locations to a dictionary for later access
    """

    def __init__(self, data_fp, metadata_fn = "consolidated_metadata.csv"):
        """
        takes in a row of a pandas dataframe and returns the class of the humanoid in the dataframe

        data_fp : location of the folder in which the metadata csv file is located
        metadata_fn : name of the metadata csv file
        """
        metadata_fp = os.path.join(data_fp, metadata_fn)
        self.fp = data_fp
        self.df = pd.read_csv(metadata_fp)
        # Add modified dataset if it exists
        modified_fp = os.path.join(data_fp, "modified_dataset", "metadata.csv")
        alt_modified_fp = os.path.join(data_fp, "metadata.csv")
        if os.path.exists(modified_fp):
            df_mod = pd.read_csv(modified_fp)
            self.df = pd.concat([self.df, df_mod], ignore_index=True)
        elif os.path.exists(alt_modified_fp):
            df_mod = pd.read_csv(alt_modified_fp)
            self.df = pd.concat([self.df, df_mod], ignore_index=True)
        self.unvisited = self.df.index.to_list()
        self.visited = []

    def reset(self):
        """
        reset list of humanoids
        """
        self.unvisited = self.df.index.to_list()
        self.visited = []

    def get_random(self):
        """
        gets and returns a random humanoid object (without replacement)
        """
        if len(self.unvisited) == 0:
            raise ValueError("No humanoids remain")
        # index = random.randint(0, (len(self.unvisited)-1))  # Technically semirandom
        index = random.choice(self.unvisited)
        # h_index = self.unvisited.pop(index)
        self.unvisited.remove(index)
        self.visited.append(index)

        datarow = self.df.iloc[index]

        state = datarow_to_state(datarow)

        humanoid = Humanoid(fp=datarow['Filename'],
                            state=state)
        return humanoid

    def get_scenario(self):
        """
        Randomly selects a scenario and returns (left_humanoid, right_humanoid, scenario_number, scenario_desc)
        """
        import random
        # Define scenario dictionary: scenario_number: (left_type, right_type)
        scenarios = {
            0: ("human", "zombie"),
            1: ("zombie", "human"),
            2: ("human", "human"),
            3: ("zombie", "zombie"),
            4: ("injured", "zombie"),
            5: ("zombie", "injured"),
            6: ("injured", "human"),
            7: ("human", "injured"),
            8: ("injured", "injured"),
            9: ("corpse", "human"),
            10: ("human", "corpse"),
            11: ("corpse", "zombie"),
            12: ("zombie", "corpse"),
            13: ("corpse", "corpse"),
            14: ("corpse", "injured"),
            15: ("injured", "corpse"),
        }
        scenario_number = random.choice(list(scenarios.keys()))
        left_type, right_type = scenarios[scenario_number]
        scenario_desc = (left_type, right_type)

        left_idx = None
        right_idx = None
        def get_random_of_type(h_type, side=None):
            nonlocal left_idx, right_idx
            if h_type == 'human':
                candidates = self.df[(self.df['Class'] == 'Default') & (self.df['Injured'] == False)]
            elif h_type == 'injured':
                candidates = self.df[(self.df['Class'] == 'Default') & (self.df['Injured'] == True)]
            elif h_type == 'zombie':
                candidates = self.df[(self.df['Class'] == 'Zombie') & (self.df['Injured'] == False)]
            elif h_type == 'corpse':
                candidates = self.df[(self.df['Class'] == 'Zombie') & (self.df['Injured'] == True)]
            else:
                raise ValueError(f"Unknown type: {h_type}")
            print(f"[DEBUG] {side} candidates count: {len(candidates)}")
            print(f"[DEBUG] {side} candidates sample: {candidates['Filename'].head().tolist()}")
            if len(candidates) == 0:
                raise ValueError(f"No candidates for type {h_type}")
            idx = random.choice(candidates.index)
            datarow = self.df.loc[idx]
            if side == 'Left':
                left_idx = idx
            elif side == 'Right':
                right_idx = idx
            state = datarow_to_state(datarow)
            debug_msg = f"[DEBUG] {side if side else ''} image filename: {datarow['Filename']} | Class: {datarow['Class']} | Injured: {datarow['Injured']} | State: {state}"
            print(debug_msg)
            return Humanoid(fp=datarow['Filename'], state=state)

        left = get_random_of_type(left_type, side='Left')
        right = get_random_of_type(right_type, side='Right')

        # Define scenario_humanoid_attributes dictionary
        scenario_humanoid_attributes = {
            'left_humanoid1': {'type': '', 'status': '', 'role': ''},
            'left_humanoid2': {'type': '', 'status': '', 'role': ''},
            'left_humanoid3': {'type': '', 'status': '', 'role': ''},
            'right_humanoid1': {'type': '', 'status': '', 'role': ''},
            'right_humanoid2': {'type': '', 'status': '', 'role': ''},
            'right_humanoid3': {'type': '', 'status': '', 'role': ''},
        }

        def assign_role(h_type, status):
            h_type_clean = h_type.strip().lower()
            if h_type_clean == 'zombie' or h_type_clean == 'corpse':
                return ''
            # Human or injured
            role_index = random.randint(0, 9)
            if role_index <= 3:
                base_role = "Civilian"
            elif role_index <= 5:
                base_role = "Child"
            elif role_index <= 7:
                base_role = "Doctor"
            elif role_index == 8:
                base_role = "Militant"
            else:
                base_role = "Police"
            if status == 'injured':
                return f"Injured {base_role}"
            else:
                return base_role

        def fill_humanoid_attributes(prefix, datarow):
            count_col = "HumanoidCount"
            class_col = "Class"
            injured_col = "Injured"
            count = int(str(datarow.get(count_col, '1')).replace('"',''))
            classes = str(datarow.get(class_col, '')).split('|')
            injureds = str(datarow.get(injured_col, '')).split('|')
            for i in range(1, 4):
                key = f"{prefix.lower()}_humanoid{i}"
                if i <= count:
                    h_type = classes[i-1].strip() if i-1 < len(classes) else ''
                    status = 'injured' if (i-1 < len(injureds) and injureds[i-1].strip().lower() == 'true') else 'healthy'
                    scenario_humanoid_attributes[key]['type'] = h_type
                    scenario_humanoid_attributes[key]['status'] = status
                    scenario_humanoid_attributes[key]['role'] = assign_role(h_type, status)
                else:
                    scenario_humanoid_attributes[key] = {'type': '', 'status': '', 'role': ''}

        left_datarow = self.df.loc[left_idx]
        right_datarow = self.df.loc[right_idx]
        fill_humanoid_attributes('Left', left_datarow)
        fill_humanoid_attributes('Right', right_datarow)

        print(f"[SCENARIO ATTRIBUTES] {scenario_humanoid_attributes}")
        return left, right, scenario_number, scenario_desc, scenario_humanoid_attributes


# can be customized
def datarow_to_state(datarow):
    """
    takes in a row of a pandas dataframe and returns the class of the humanoid in the dataframe

    datarow : row of the metadata dataframe
    """
    img_path = datarow['Filename']
    img_class = datarow['Class']
    img_injured = datarow['Injured']
    # img_gender = datarow['Gender']
    # img_item = datarow['Item']
    # state = ""
    if img_class == 'Default':
        state = State.HEALTHY.value
        if img_injured:
            state = State.INJURED.value
    else:
        state = State.ZOMBIE.value
        if img_injured:
            state = State.CORPSE.value
    return state
