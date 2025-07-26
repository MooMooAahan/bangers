import random
import pandas as pd
from gameplay.humanoid import Humanoid
from gameplay.enums import State
import os
from gameplay.image import Image
import torchvision.transforms as transforms


class DataParser(object):
    """
    Parses the input data photos and assigns their file locations to a dictionary for later access
    """

    def __init__(self, data_fp, metadata_fn = "metadata.csv"):
        """
        takes in a row of a pandas dataframe and returns the class of the humanoid in the dataframe

        data_fp : location of the folder in which the metadata csv file is located
        metadata_fn : name of the metadata csv file
        """
        metadata_fp = os.path.join(data_fp, metadata_fn)
        self.fp = data_fp
        self.df = pd.read_csv(metadata_fp)

        # Add modified dataset if 
        # alt_modified_fp = os.path.join(data_fp, "metadata.csv")
        # df_mod = pd.read_csv(alt_modified_fp)
        # self.df = pd.concat([self.df, df_mod], ignore_index=True)
        self.unvisited = self.df.index.to_list()
        self.visited = []
        # Standardize 'Class' and 'Injured' columns for consistent filtering
        self.df['Class'] = self.df['Class'].astype(str).str.strip().str.capitalize()

    def reset(self):
        """
        reset list of humanoids
        """
        self.unvisited = self.df.index.to_list()
        self.visited = []

    def get_random(self, side): # either left or right side
        """
        gets and returns a random Image object (without replacement)
        """
        if len(self.unvisited) == 0:
            raise ValueError("No images remain")
        # index = random.randint(0, (len(self.unvisited)-1))  # Technically semirandom
        # h_index = self.unvisited.pop(index)

        #TODO: make sure that image selected is from the correct side. may be able to alter this when creating final dataset, and images are on separate sides?
        
        # select a random index from unvisited that matches the side
        index = random.choice(self.unvisited)
        if side == 'left':
            while self.df.iloc[index]['Side'] != 'Left':
                index = random.choice(self.unvisited)
        elif side == 'right':
            while self.df.iloc[index]['Side'] != 'Right':
                index = random.choice(self.unvisited)
        elif side == 'random':
            pass
        else:
            raise ValueError("Invalid side")
        # while side != self.df.iloc[index]['Side']:
        #     index = random.choice(self.unvisited)
        # remove the index from unvisited and add to visited
        self.unvisited.remove(index)
        self.visited.append(index)

        datarow = self.df.iloc[index]

        image = Image(datarow)
        return image
    
    def get_augmented_transforms(self):
        """
        Returns data augmentation transforms for training
        Designed for character images with backgrounds - 5x data multiplication
        """
        return transforms.Compose([
            transforms.Resize((512, 512)),  # Standardize input size
            transforms.RandomHorizontalFlip(p=0.5),  # Mirror scenes - very effective
            transforms.RandomRotation(degrees=10),   # Slight camera angle variation
            transforms.ColorJitter(
                brightness=0.3,    # Day/night lighting conditions
                contrast=0.2,      # Weather/visibility variations
                saturation=0.2,    # Environmental effects
                hue=0.1           # Slight color shifts
            ),
            transforms.RandomResizedCrop(512, scale=(0.9, 1.0)),  # Slight zoom variation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_validation_transforms(self):
        """Returns transforms for validation/testing (no augmentation)"""
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def create_enhanced_label(self, row):
        """
        Create enhanced 21-class label from metadata row
        Args:
            row: DataFrame row with Class, Injured, Role columns
        Returns:
            Enhanced class string like 'zombie_police'
        """
        # Parse class (Zombie or Default)
        class_val = row.get('Class', 'Default')
        injured_val = row.get('Injured', 'False')
        role_val = row.get('Role', 'Civilian')
        
        # Convert class to status
        if str(class_val).lower() == 'zombie':
            if str(injured_val).lower() == 'true':
                status = 'corpse'  # Injured zombie = corpse
            else:
                status = 'zombie'  # Healthy zombie
        else:  # Default
            if str(injured_val).lower() == 'true':
                status = 'injured'  # Injured human
            else:
                status = 'healthy'  # Healthy human
        
        # Clean up occupation
        occupation = str(role_val).lower().strip()
        
        # Map common variations to standard occupations
        occupation_map = {
            'blank': 'civilian',
            'unknown': 'civilian',
            '': 'civilian',
            'police': 'police',
            'doctor': 'doctor',
            'child': 'child',
            'civilian': 'civilian',
            'militant': 'militant'
        }
        
        occupation = occupation_map.get(occupation, 'civilian')
        
        # Create enhanced class using Humanoid helper
        return Humanoid.create_enhanced_class(status, occupation)


