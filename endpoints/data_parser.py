import random
import pandas as pd
from gameplay.humanoid import Humanoid
from gameplay.enums import State
import os
from gameplay.image import Image


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
        return datarow


