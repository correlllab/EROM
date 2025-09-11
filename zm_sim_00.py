### Standard ### 
from collections import deque
from random import random

### Special ### 
import numpy as np

### ASPIRE::PDDLStream ### 
from aspire.symbols import ObjPose, GraspObj, extract_pose_as_homog
from aspire.env_config import env_var, env_sto
from aspire.BlocksTask import set_blocks_env

### Local ### 
from env_config import set_experiment_env
from env_config import KNOWN_BLOCKS



########## SIMULATION CLASSES ######################################################################

class Engine:
    """ Shit Happens """
    def __init__( self ):
        self.objs = KNOWN_BLOCKS()
        self.prob = {
            "ActionFailure" : 0.10,
        }


    def pose_above( self, target : GraspObj ) -> ObjPose:
        """ Get the stacking pose above the `target` """
        pose = extract_pose_as_homog( target )
        pose[2,3] += env_var("_BLOCK_SCALE")
        return ObjPose( pose.copy() )


    def stack_A_onto_B( self, A : GraspObj, B : GraspObj, factList : list[tuple] ) -> list[tuple]:
        """ Add stacked states """
        if random() < self.prob["ActionFailure"]:
            pass
        else:
            abovPose = self.pose_above(B)
            A.pose = abovPose
            factList.extend( [
                ('GraspObj' , A.label, A.pose.copy() ),
                ('Supported', A.label, B.label ),
                ('Blocked'  , B.label ),
            ] )
            return factList
    

    def negate_fact( self, negFct : tuple, factList : list[tuple] ) -> list[tuple]:
        """ Remove a fact from the list and return the list """
        rtnLst = list()
        for fact in factList:
            if fact[0] == negFct[0]:
                if (negFct[0] == "Supported") or (negFct[0] == "GraspObj"):
                    if (fact[1] == negFct[1]):
                        continue
            rtnLst.append( fact )
        return rtnLst
    

    def negate_many( self, negLst : list, factList : list[tuple] ) -> list[tuple]:
        """ Serially negate a list of facts """
        for negFct in negLst:
            factList = self.negate_fact( negFct, factList )
        return factList
    

    def unstack_A_from_B( self, A : GraspObj, B : GraspObj, AdstPose : np.ndarray, factList : list[tuple] ) -> list[tuple]:
        """ Unstack `A` and set it on the 'table' """
        poseA    = ObjPose( np.array( AdstPose ) )
        factList = self.negate_many( [
            ('GraspObj' , A.label ),
            ('Supported', A.label ),
            ('Blocked'  , B.label ),
        ], factList )
        A.pose = poseA
        factList.extend( [
            ('GraspObj' , A.label, A.pose.copy() ),
            ('Supported', A.label, 'table' ),
        ] )
        return factList
    

    def place_A( self, A : GraspObj, AdstPose : np.ndarray, factList : list[tuple] ) -> list[tuple]:
        poseA    = ObjPose( np.array( AdstPose ) )
        factList = self.negate_many( [('GraspObj' , A.label ),], factList )
        A.pose = poseA
        factList.extend( [('GraspObj' , A.label, A.pose.copy() ),] )
        return factList



class Solver:
    """ Who needs PDDL? """

    def set_sim_env( self ):
        """ Set necessary params """
        env_sto( "_GOAL_SIM" ,
            ( 'and',
                ('GraspObj', 'grnBlock', self.poses[0] ),
                ('GraspObj', 'redBlock', self.poses[1] ), 
                ('GraspObj', 'bluBlock', self.poses[2] ), 
            )        
        )


    def __init__( self ):
        """ Get ready to solve """
        self.poses = list()
        pose = np.eye(4)
        pose[2,3] = 0.5 * env_var("_BLOCK_SCALE")
        self.poses.append( ObjPose( pose.copy() ) )
        pose[2,3] += env_var("_BLOCK_SCALE")
        self.poses.append( ObjPose( pose.copy() ) )
        pose[2,3] += env_var("_BLOCK_SCALE")
        self.poses.append( ObjPose( pose.copy() ) )
        set_blocks_env()
        set_experiment_env()
        self.set_sim_env()


    def ground_facts( self, objs : list[GraspObj] ):
        """ Set facts from the object list """
        pass



    def solve( self, facts : tuple, goal : tuple ):
        pass






class SimPlanner:
    """ Dice Roll Planner, w/o PDLS """
    def __init__( self ):
        self.goal = None
        self.plan = None
        