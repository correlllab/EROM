

########## SIMULATION CLASSES ######################################################################

class SimBlock:
    """ Container class for a Block """
    count = 0

    def __init__( self, label = None, pose = None, stacked = False , blocked = False ):
        """ Minimal state to represent a stacked block """
        SimBlock.count += 1
        self.id   : int   = SimBlock.count
        self.label: str   = label
        self.pose : float = pose


    def __repr__( self ):
        """ Print state """
        return f"({self.label} @ {self.pose}), id: {self.id}"

    
    def copy( self ):
        """ Make a copy w a new ID """
        rtnObj = SimBlock()
        rtnObj.label   = self.label   
        rtnObj.pose    = self.pose    
        return rtnObj