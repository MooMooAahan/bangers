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

        def get_random_of_type(h_type):
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
            if len(candidates) == 0:
                raise ValueError(f"No candidates for type {h_type}")
            idx = random.choice(candidates.index)
            datarow = self.df.loc[idx]
            state = datarow_to_state(datarow)
            return Humanoid(fp=datarow['Filename'], state=state)

        left = get_random_of_type(left_type)
        right = get_random_of_type(right_type)
        print(f"[SCENARIO DEBUG] Scenario {scenario_number}: left={left_type} ({left.fp}), right={right_type} ({right.fp})")
        return left, right, scenario_number, scenario_desc


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
