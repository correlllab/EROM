### Standard ### 
import os, json
from collections import deque
from random import random, choice
from enum import Enum
from pprint import pprint
from copy import deepcopy
from dataclasses import dataclass, field

### Special ### 
import numpy as np

### ASPIRE::PDDLStream ### 
from aspire.symbols import ObjPose, GraspObj, euclidean_distance_between_symbols
from aspire.env_config import env_var, env_sto
from aspire.BlocksTask import set_blocks_env

### Local ### 
from env_config import set_experiment_env
# from env_config import KNOWN_BLOCKS
from ya_SimClasses import SimBlock


@dataclass
class Action:
    """ Move `heldBlc` from `bgnPose` to `endPose` """
    bgnPose : float = -1.0
    endPose : float = -1.0
    heldBlc : str   = None



class SimplePlanner:
    """ Cheap Planner for Simulation """
    def __init__( self ):
        self.goals = {
            "RGB" : ["GRN", "RED", "BLU"],
            "RBW" : ["WHT", "RED", "BLK"],
        }
        self.poses            = [1.0, 2.0, 3.0,]
        self.goal : list[str] = None


    @staticmethod
    def block_at_pose( state : list[SimBlock], pose : float ):
        """ Return the class of the block at the pose, Otherwise retun None """
        dMin = 6e10
        lMin = None
        for block in state:
            d = abs( pose - block.pose )
            if d <= env_var("_PLACE_XY_ACCEPT"):
                if d < dMin:
                    dMin = d
                    lMin = block.label
        return lMin
    

    @staticmethod
    def random_pose():
        """ Return a pose outside any of the starting or goal poses, Do NOT check for collisions! """
        return (16 + int(random()*1000)) * 1.0


    def plan( self, setting : str, state : list[SimBlock] ):
        """ Get a plan for the given state """
        self.goal = self.goals[ setting ] 
        ## Step 0: Ground the State ##
        compare = deque()
        for target in self.poses:
            compare.append( SimplePlanner.block_at_pose( state, target ) )
        ## Step 1: Check goal ##
        goalMet = True
        for i in range( len( self.goal ) ):
            pass # FIXME: START HERE - CHECK FOR THE GOAL
            
        ## Step 2: Check for corrections ##
        ## Step 3: Build remainder ##

