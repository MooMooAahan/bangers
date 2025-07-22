class Image(object):
    """
    Stores all metadata for an image and references to Humanoid objects.
    """
    def __init__(self, datarow):
        """
        datarow: pandas Series (row from DataParser.df)
        humanoids: list of Humanoid objects (or a single Humanoid)
        """
        # Store all metadata fields as attributes
        for col in datarow.index:
            setattr(self, col, datarow[col])
        # Store humanoid(s)
        self.humanoids = humanoids if isinstance(humanoids, list) else [humanoids]

        # get information 
        state = datarow_to_state(datarow)



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



    def __repr__(self):
        return f"<ImageData Filename={getattr(self, 'Filename', None)} Humanoids={self.humanoids}>"