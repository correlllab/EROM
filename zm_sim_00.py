### Standard ### 
from collections import deque

### Special ### 
import numpy as np

### ASPIRE::PDDLStream ### 
from aspire.symbols import ObjPose, GraspObj
from aspire.env_config import env_var
from aspire.BlocksTask import set_blocks_env

### Local ### 
from env_config import set_experiment_env
from env_config import KNOWN_BLOCKS



########## SIMULATION CLASSES ######################################################################

class Engine:
    """ Shit Happens """
    def __init__( self ):
        self.objs = KNOWN_BLOCKS()



class Solver:
    """ Who needs PDDL? """
    def __init__( self ):
        pose0      = np.eye(4)
        pose0[2,3] = 0.5 * env_var("_BLOCK_SCALE")
        pose0      = np.eye(4)
        self.poses = [
            ObjPose(  )
        ]
        self.goal  = None
        self.facts = list()



class SimPlanner:
    """ Dice Roll Planner, w/o PDLS """
    def __init__( self ):
        self.goal = None
        self.plan = None
        set_blocks_env()
        set_experiment_env()