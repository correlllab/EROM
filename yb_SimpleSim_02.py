########## INIT ####################################################################################

### Standard ### 
import os, json
from collections import deque
from random import random, choice
from enum import Enum
from pprint import pprint
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Deque

### Special ### 
import numpy as np
from scipy.stats import lognorm

### ASPIRE::PDDLStream ### 
from aspire.symbols import ObjPose, GraspObj, euclidean_distance_between_symbols
from aspire.env_config import env_var, env_sto
from aspire.BlocksTask import set_blocks_env

### Local ### 
from env_config import set_experiment_env
from analysis_Utils import DiceBinary_CDF, Loc
from ya_SimClasses import SimBlock


""" ########## DEV_PLAN ############################################################################
[Y] Support Functions / Classes
    [Y] Perception
        [Y] Roll Confusion
        [Y] Roll Pose Error
        [Y] Roll Hallucination
    [Y] Planning
        [Y] Roll Planning Success: P( Plan | N_halluc )  
        [Y] Solve for Plan
    [Y] Action
        [Y] Roll Action Outcome: P( Success | Pose Error )
            [Y] Success: Update Actual State
            [Y] Failure: Roll tower destruction

[ ] Simulation Loop
    [ ] Perception
        [ ] Build Perceived State
            [ ] Roll Confusion
            [ ] Roll Pose Error
        [ ] Roll Hallucination
    [ ] Planning
        [ ] Roll Planning Success: P( Plan | N_halluc )  
        [ ] Solve for Plan
    [ ] Action
        [ ] Roll Action Outcome: P( Success | Pose Error )
            [ ] Success: Update Actual State
            [ ] Failure: Roll tower destruction
    [ ] While NOT solved, ^^^ LOOP ^^^

[ ] Iterate Datasets
[ ] Iterate Scenarios

* Issues:
    - Seems like N_halluc would ALSO influence action success?

"""

########## SIMULATION CLASSES ######################################################################

##### Action ##############################################################

@dataclass
class Action:
    """ Move `heldBlc` from `bgnPose` to `endPose` """
    bgnPose : float = -1.0
    endPose : float = -1.0
    heldBlc : str   = None



##### Engine ##############################################################
_SETTINGS_PATH = "$HOME/EROM/json/sim_settings.json"


class Engine:
    """ Simulate reality in the cheapest way possible """
    def __init__( self ):
        """ Get ready to simulate! """
        self.settings = dict()
        with open( os.path.expandvars( _SETTINGS_PATH ), 'r' ) as f:
            self.settings = json.load(f)
        self.actualState: Deque[SimBlock] = deque()
        self.initPoses  : list[float]     = [4.0, 5.0, 6.0,]
        self.actionCDF  : DiceBinary_CDF  = None
        self.init_action_cdf()


    ##### Probabilistic Outcomes #################

    @staticmethod
    def roll_probs_as_odds( probs : list[float] ) -> int:
        """ Return the index of the outcome, given a collection of ordered odds """
        total = 0.0
        scale = sum( probs )
        odds  = np.zeros( (len( probs ),) )
        for i, prob in enumerate( probs ):
            total += prob
            odds[i] = total
        odds /= scale # This should have summed to 1, but Be Prepared
        roll = random()
        for i, odd in enumerate( odds ):
            if roll < odd:
                return i
        return len( probs )-1
    

    @staticmethod
    def roll_dict_as_odds( table : dict[str,float] ):
        """ Roll a dictionary of odds, Return key associated with rolled event """
        keys = deque()
        odds = deque()
        for k, v in table.items():
            keys.append(k)
            odds.append(v)
        res = Engine.roll_probs_as_odds( odds )
        return keys[ res ]


    def roll_confusion( self, dataset : str, actual : str ):
        """ What class will the robot see? """
        confMatx = np.array( self.settings[ dataset ]["confMatx"] )
        if actual in self.settings[ dataset ]["classes"]:
            ndx = self.settings[ dataset ]["classes"].index( actual )
            row = confMatx[ ndx, : ]
            res = Engine.roll_probs_as_odds( row )
            return self.settings[ dataset ]["classes"][ res ]
        raise KeyError( f"{actual} is NOT a valid label for the {dataset} dataset!" )

    
    def roll_avg_pose_error( self, dataset : str, test : str ) -> float:
        """ Roll from the (average) pose error for this `dataset`::`test` """
        return lognorm.rvs( 
            loc   = self.settings[ dataset ][ test ]["poseErr"]["location"] * 1.0, 
            shape = self.settings[ dataset ][ test ]["poseErr"]["shape"], 
            scale = self.settings[ dataset ][ test ]["poseErr"]["scale"]
        )
    

    @staticmethod
    def extract_int( expr : str ) -> int:
        iStr = ""
        for char in expr:
            if char.isdigit():
                iStr += char
        return int( iStr )
    

    def roll_hallucination( self, dataset : str, test : str ) -> int:
        """ Get the number of hallucinated blocks """
        keys  = list( self.settings[ dataset ][ test ].keys() )
        probs = dict()
        for key in keys:
            if (" Halluc)" in key) and ('|' not in key):
                probs[ key ] = self.settings[ dataset ][ test ][ key ]
        ans = Engine.roll_dict_as_odds( probs )
        return Engine.extract_int( ans )
    

    def roll_plan_success( self, dataset : str, test : str, N_halluc : int ) -> bool:
        """ Return whether a plan resulted from symbols given a number of hallucinations """
        keys  = list( self.settings[ dataset ][ test ].keys() )
        probs = dict()
        for key in keys:
            if (" Halluc)" in key) and ('|' in key):
                probs[ key ] = self.settings[ dataset ][ test ][ key ]
        for k, v in probs.items():
            if Engine.extract_int(k) == N_halluc:
                roll = random()
                if roll <= v:
                    return True
                else:
                    return False
        raise RuntimeError( f"Did NOT find a plan probability for {N_halluc} hallucinations!" )
    

    def init_action_cdf( self, cdfPath : str = None):
        """ Rebuild the CDF from the JSON file """
        if cdfPath is None:
            self.actionCDF = DiceBinary_CDF.load( Loc._JSON_PATH["Failure-v-Err_CDF"] )
        else:
            self.actionCDF = DiceBinary_CDF.load( cdfPath )


    def roll_action_success( self, avgPoseErr : float ) -> bool:
        """ Use the CDF to determine whether a failure occurred """
        return not self.actionCDF.sample_outcome( avgPoseErr )
    

    def roll_tower_desctruction( self, dataset : str, test : str ):
        """ How many blocks did the tower lose on a failed action? """
        table : dict[str,float] = self.settings[ dataset ][ test ]["actFailLostSteps"]
        if len( table ):
            ans = Engine.roll_dict_as_odds( table )
            return Engine.extract_int( ans )
        return 0
    

    ##### Deterministic Outcomes #################

    def reset_blocks( self, dataset : str ):
        """ Get ready for a new episode! """
        self.actualState = deque()
        for i, pose in enumerate( self.initPoses ):
            self.actualState.append( SimBlock(
                label = self.settings[ dataset ]["labels"][i],
                pose  = pose
            ) )


    @staticmethod
    def block_at_pose( state : list[SimBlock], pose : float ) -> SimBlock:
        """ Return the class of the block at the pose, Otherwise retun None """
        dMin = 6e10
        bMin = None
        for block in state:
            d = abs( pose - block.pose )
            if d <= env_var("_PLACE_XY_ACCEPT"):
                if d < dMin:
                    dMin = d
                    bMin = block
        return bMin


    def apply_action( self, action : Action ) -> bool:
        """ Change the true state for an action that succeeds """
        bgnPose = action.bgnPose
        target  = Engine.block_at_pose( self.actualState, bgnPose )
        if target is not None:
            endPose = action.endPose
            target.pose = endPose
            return True
        return False
    


##### Planner #############################################################


class SimplePlanner:
    """ Cheap Planner for Simulation """

    goals = {
        "RGB" : ["GRN", "RED", "BLU"],
        "RBW" : ["WHT", "RED", "BLK"],
    }

    def __init__( self ):
        self.poses            = [1.0, 2.0, 3.0,]
        self.goal : list[str] = None


    @staticmethod
    def copy_state( state : list[SimBlock] ) -> list[SimBlock]:
        """ Deep copy of `state` """
        rtnStt = deque()
        for block in state:
            rtnStt.append( block.copy() )
        return list( rtnStt )


    @staticmethod
    def label_at_pose( state : list[SimBlock], pose : float ):
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
    def get_block_by_name( state : list[SimBlock], label : float ):
        """ Fetch the named block """
        for block in state:
            if block.label == label:
                return block
        return None
    

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
            compare.append( SimplePlanner.label_at_pose( state, target ) )
        
        ## Step 1: Check goal ##
        goalMet = True
        blcDiff = deque()
        for i in range( len( self.goal ) ):
            if self.goal[i] != compare[i]:
                goalMet = False
                blcDiff.append( compare[i] )
            else:
                blcDiff.append( True )
        blcDiff = list( blcDiff )

        if goalMet:
            return list()
            
        rtnPln : Deque[Action] = deque()
        
        ## Step 2: Check for corrections ##
        wrong   = False
        nuState = SimplePlanner.copy_state( state )
        for i in range( len( self.goal ) ):
            if (wrong or (blcDiff[i] != True)) and (blcDiff[i] is not None):
                wrong = True
                rtnPln.appendleft( Action(
                    bgnPose = self.poses[i],
                    endPose = SimplePlanner.random_pose(),
                    heldBlc = blcDiff[i]
                ) )
                nuBlc = SimplePlanner.get_block_by_name( nuState, blcDiff[i] )
                nuBlc.pose = rtnPln[0].endPose
                blcDiff[i] = None
        
        ## Step 3: Build remainder ##
        for i in range( len( self.goal ) ):
            if blcDiff[i] == None:
                nuBlc = SimplePlanner.get_block_by_name( nuState, self.goal[i] )
                rtnPln.append( Action(
                    bgnPose = nuBlc.pose,
                    endPose = self.poses[i],
                    heldBlc = self.goal[i]
                ) )

        return list( rtnPln )


########## MAIN ####################################################################################
