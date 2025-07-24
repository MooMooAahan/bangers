from gameplay.humanoid import Humanoid
from gameplay.enums import State
import pandas as pd

class Image(object):
    """
    Stores all metadata for an image and references to up to two Humanoid objects.
    """
    def __init__(self, datarow):
        self.datarow = datarow  # Store the original datarow for later access
        # Store all metadata fields as attributes
        for col in datarow.index:
            val = datarow[col]
            if col == "Filename" and isinstance(val, str):
                val = val.strip().strip('"').strip("'")
            setattr(self, col, val)

        # Parse humanoid count
        try:
            self.humanoid_count = int(datarow['HumanoidCount'])
        except Exception:
            self.humanoid_count = 0

        # Parse fields for up to two humanoids
        def split_or_none(val):
            if pd.isna(val) or val == "":
                return [None, None]
            parts = str(val).split('|')
            if len(parts) == 1:
                return [parts[0], None]
            elif len(parts) >= 2:
                return [parts[0], parts[1]]
            return [None, None]

        # Extract all relevant fields
        classes = split_or_none(datarow['Class'])
        roles = split_or_none(datarow['Role'])
        injureds = split_or_none(datarow['Injured'])
        genders = split_or_none(datarow['Gender'])
        items = split_or_none(datarow['Item'])

        # Helper to get state from class/injured
        def get_state(cls, inj):
            if cls is None or cls == "":
                return None
            if str(cls).lower() == "default":
                return State.INJURED.value if str(inj).lower() == "true" else State.HEALTHY.value
            elif str(cls).lower() == "zombie":
                return State.CORPSE.value if str(inj).lower() == "true" else State.ZOMBIE.value
            return None

        # Create up to two humanoids
        self.humanoids = []
        for i in range(2):
            if self.humanoid_count > i and classes[i] not in (None, "", "nan"):
                state = get_state(classes[i], injureds[i])
                self.humanoids.append(
                    Humanoid(fp=str(datarow['Filename']).strip(), state=state, role=roles[i])
                )
            else:
                self.humanoids.append(None)  # Null if not present

    def __repr__(self):
        return f"<ImageData Filename={getattr(self, 'Filename', None)} HumanoidCount={self.humanoid_count} Humanoids={self.humanoids}>"