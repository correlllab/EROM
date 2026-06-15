from random import random, choice

import numpy as np

########## METRICS + SETTINGS ######################################################################

confMatrices = {

    "RGB": np.array( [[0.81077 , 0.0087336, 0.034934, 0.14556 ,],
                      [0.018576, 0.74149  , 0.17028 , 0.069659,],
                      [0.037179, 0.12051  , 0.76282 , 0.079487,],
                      [0.041032, 0.069949 , 0.034388, 0.85463 ,],] ), 

    "RBW": np.array( [[0.73793 , 0.046552, 0.055172 , 0.16034,],
                      [0.045388, 0.75988 , 0.0087848, 0.18594,],
                      [0.027451, 0.015686, 0.72745  , 0.22941,],
                      [0.059389, 0.034934, 0.10175  , 0.80393,],] ),
}

names = {
    "RGB": {
        "labels" : ["RED", "GRN", "BLU",],
        "classes": ["RED", "GRN", "BLU", "NOTHING",],
    },
    "RBW": {
        "labels" : ["RED", "BLK", "WHT",],
        "classes": ["RED", "BLK", "WHT", "NOTHING",],
    },
}


def roll_sensed_class( actual : str, dataset : str ):
    """ Return the sensed class given the `actual` class, based on the associated row of the confusion matrix """
    classes: list[str] = names[ dataset ]["classes"]
    index  : int       = 6e10
    try:
        index = classes.index( actual )
    except ValueError:
        print( f"{actual} is NOT an object class!" )
        return None
    row  = confMatrices[ dataset ][ index ]
    odds = list()
    totl = 0.0
    for col in row:
        totl += col
        odds.append( totl )
    uniform = random()
    for i, bound in enumerate( odds ):
        if uniform <= bound:
            return classes[i]
    return classes[-1] 



########## SIMULATION CLASSES ######################################################################

##### Blocks ############################################################## 
_ONE_START = 4.0
_TWO_START = 5.0
_THR_START = 6.0

_ONE_TARGT = 1.0
_TWO_TARGT = 2.0
_THR_TARGT = 3.0

_NAMES   = ["RED", "GRN", "BLU",]
_CLASSES = list( _NAMES ) + ["NOTHING",]

class SimBlock:
    """ Container class for a Block """
    count = 0

    def __init__( self, label = None, pose = None ):
        """ Minimal state to represent a stacked block """
        SimBlock.count += 1
        self.id   : int   = SimBlock.count # "Unique" Identifier
        self.label: str   = label # -------- Class of the block
        self.pose : float = pose # --------- Pose as float: {Whole: Pose}.{Frac: Distance from Ideal [m]}


    def __repr__( self ):
        """ Print state """
        return f"({self.label} @ {self.pose}), id: {self.id}"

    
    def copy( self ):
        """ Make a copy w a new ID """
        rtnObj = SimBlock()
        rtnObj.label = self.label   
        rtnObj.pose  = self.pose    
        return rtnObj
    


##### Transition Model ####################################################

class Engine:
    """ Shit Happens """

    ##### Static Methods #########################

    ## Class Vars ##
    poses  = set( [i*1.0 for i in range( 1,7 )] ) # Why is this here?
    bignum = 1e5 # -------------------------------- ASSUMPTION: WE DO NOT NEED MORE THAN `bignum` POSES! 


    ##### Static Methods ##################################################

    @staticmethod
    def init_blocks() -> list[SimBlock]:
        """ Get all the blocks in the scene """
        rtnLst = list()
        poses  = [_ONE_START, _TWO_START, _THR_START,]
        for i, name in enumerate( _NAMES ):
            rtnLst.append( SimBlock( label = name, pose = poses[i] ) )
        return rtnLst
    

    ##### General Methods #################################################

    def __init__( self ):
        """ Setup a New Episode """
        self.objs = Engine.init_blocks()

    
    ##### Perception ######################################################

    def noisy_sense( self ):
        """ Get noisy readings of all the objects in the World """
        # Copy #
        rtnLst : list[SimBlock] = list()
        for obj in self.objs:
            rObj = obj.copy()
            rtnLst.append( rObj )

        # Handle Class Confusion #
        for obj in rtnLst:
            if random() < self.params["classConfuse"]:
                oldLbl = obj.label
                lblNew = obj.label
                while oldLbl == lblNew:
                    lblNew = choice( _NAMES )
                obj.label = lblNew

        return rtnLst
        
