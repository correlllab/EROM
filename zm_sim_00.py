### Standard ### 
import os
from collections import deque
from random import random, choice
from enum import Enum

### Special ### 
import numpy as np

### ASPIRE::PDDLStream ### 
from aspire.symbols import ObjPose, GraspObj, extract_pose_as_homog, euclidean_distance_between_symbols
from aspire.env_config import env_var, env_sto
from aspire.BlocksTask import set_blocks_env

### Local ### 
from env_config import set_experiment_env
from env_config import KNOWN_BLOCKS



########## SIMULATION CLASSES ######################################################################

class SimBlock:
    """ Container class for a Block """
    count = 0

    def __init__( self, label = None, pose = None, stacked = False , blocked = False ):
        """ Minimal state to represent a stacked block """
        SimBlock.count += 1
        self.id      = SimBlock.count
        self.label   = label
        self.pose    = pose

    def __repr__( self ):
        """ Print state """
        return f"({self.label} @ {self.pose}), id: {self.id}"
    
    def copy( self ):
        rtnObj = SimBlock()
        rtnObj.label   = self.label   
        rtnObj.pose    = self.pose    
        return rtnObj




########## HELPER FUNCTIONS ########################################################################
_P_CONF = 0.10
_NAMES  = ["A","B","C",]
# _PLAN   = [
#     ("Place", "A", 0,),
#     ("Stack", "B", 1,),
#     ("Stack", "C", 2,),
# ] 



########## TRANSITION MODEL ########################################################################

class Engine:
    """ Shit Happens """

    ##### Static Methods ##################################################

    ## Class Vars ##
    _poses = set([i for i in range(3)])


    @staticmethod
    def roll_pose() -> int:
        return int( 3 + random() * 1000 )


    @staticmethod
    def rand_pose() -> int:
        """ Generate a random int above 2 """
        # ASSUMPTION: WE DO NOT NEED MORE THAN 1003 POSES!
        nuPose = Engine.roll_pose()
        while nuPose in Engine._poses:
            nuPose = Engine.roll_pose()
        Engine._poses.add( nuPose )
        return nuPose


    @staticmethod
    def init_blocks() -> list[SimBlock]:
        """ Get all the blocks in the scene """
        rtnLst = list()
        for name in _NAMES:
            rtnLst.append( SimBlock( name, Engine.rand_pose() ) )
        return rtnLst


    ##### General Methods #################################################

    def __init__( self ):
        """ Setup a New Episode """
        self.prob = {  "ActionFailure" : 0.10,  }
        self.objs = Engine.init_blocks()

    def report( self ):
        """ Print what is happening with the objects """
        print( "\n##### Current State of World Objects #####" )
        for obj in self.objs:
            print( obj )
        print()


    ##### Perception ######################################################

    def noisy_sense( self ):
        """ Get noisy readings of all the objects in the World """
        rtnLst = list()
        for obj in self.objs:
            rObj = obj.copy()
            # Handle Class Confusion #
            if random() < _P_CONF:
                while rObj.label == obj.label:
                    rObj.label = choice( _NAMES )
            rtnLst.append( rObj )
        return rtnLst


    ##### Actions #########################################################

    def get_obj_from_pose( self, qPose : int ):
        """ Fetch the object at the expected `pose` """
        for obj in self.objs:
            if abs(obj.pose - qPose) <= env_var("_ACCEPT_POSN_ERR"):
                return obj
        return None
    

    def place( self, actDesc : list ):
        """ Execute the "Place" action, even if the label is WRONG!, Return whether the action was successful """
        # ASSUMPTION: MOVED OBJECT EXISTS
        actName = actDesc[0]
        _       = actDesc[1] # `Engine` doesn't actually care what the label is!
        bgnPose = actDesc[2]
        endPose = actDesc[3]
        actObjc = self.get_obj_from_pose( bgnPose )

        def roll_success() -> bool:
            """ Return True if the die roll passes, Else apply failure transition and return False """
            nonlocal self, actObjc
            if random() >= self.prob["ActionFailure"]:
                return True
            else:
                print( f"ACTION FAILED: {actDesc}" )
                actObjc.pose = Engine.rand_pose()
                return False
        
        if actName == "Place":
            ## Apply Transition ##
            actObjc.pose = endPose
            return roll_success()
        else:
            raise ValueError( f"BAD DESC. for \"Place\": {actDesc}""" )


    def stack( self, actDesc : list ):
        """ Execute the "Stack" action, even if the labels are WRONG!, Return whether the action was successful """
        # ASSUMPTION: MOVED OBJECT EXISTS
        actName = actDesc[0]
        bgLabl  = actDesc[1]
        upLabl  = actDesc[2]
        bgPose  = actDesc[3]
        upPose  = actDesc[4]
        dnPose  = actDesc[5]
        bgObjct = self.get_obj_from_pose( bgPose )
        dnObjct = self.get_obj_from_pose( dnPose )

        def roll_success( forceFail : bool = False ):
            nonlocal self, bgObjct
            if (not forceFail) and (random() >= self.prob["ActionFailure"]):
                return True
            else:
                print( f"ACTION FAILED: {actDesc}" )
                bgObjct.pose = Engine.rand_pose()
                return False

        if actName == "Stack":
            ## Test Physical Plausibility ##
            if dnObjct is None:
                return roll_success( forceFail = True )
            ## Apply Transition ##
            bgObjct.pose = upPose
            return roll_success()
        else:
            raise ValueError( f"BAD DESC. for \"Stack\": {actDesc}""" )
        

    def unstack( self, actDesc : list ):
        """ Execute the "Unstack" action, even if the labels are WRONG!, Return whether the action was successful """
        # ASSUMPTION: MOVED OBJECT EXISTS
        actName = actDesc[0]
        _       = actDesc[1]
        bgnPose = actDesc[2]
        endPose = actDesc[3]
        actObjc = self.get_obj_from_pose( bgnPose )

        def roll_success( forceFail : bool = False ) -> bool:
            """ Return True if the die roll passes, Else apply failure transition and return False """
            nonlocal self, actObjc
            if (not forceFail) and (random() >= self.prob["ActionFailure"]):
                return True
            else:
                print( f"ACTION FAILED: {actDesc}" )
                actObjc.pose = Engine.rand_pose()
                return False

        if actName == "Unstack":
            ## Apply Transition ##
            actObjc.pose = endPose
            return roll_success()
        else:
            raise ValueError( f"BAD DESC. for \"Unstack\": {actDesc}""" )

########## SIMPLEST SOLVER #########################################################################

class Status( Enum ):
    """ Planner Status """
    INVALID = "INVALID"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class Solver:
    """ Who needs PDDL? """

    def set_sim_env( self ):
        """ Set necessary params """
        env_sto( "_GOAL_SIM" ,
            [ 'and',
                ['GraspObj', 'A', self.ascendingPoses[0] ],
                ['GraspObj', 'B', self.ascendingPoses[1] ], 
                ['GraspObj', 'C', self.ascendingPoses[2] ], 
            ]        
        )


    def __init__( self ):
        """ Get ready to solve """
        self.ascendingPoses = [0,1,2,]
        self.status         = Status.INVALID
        set_blocks_env()
        set_experiment_env()
        self.set_sim_env()


    @staticmethod
    def get_goal_facts( goal : list ):
        """ Get the individual facts from the `goal` """
        if goal[0] in ("and", "or"):
            rtnFcs = list()
            for item in goal[1:]:
                rtnFcs.extend( Solver.get_goal_facts( item ) )
            return rtnFcs
        elif goal[0] == "not":
            return list()
        else:
            return [goal,]
        

    @staticmethod
    def get_fact_by_pose( q : int, facts : list[list] ):
        """ Return true if the facts imply that a pose is occupied by an object """
        for fact in facts:
            if fact[0] == "GraspObj":
                if abs( q - fact[2] ) <= env_var("_ACCEPT_POSN_ERR"):
                    return fact[:]
        return None
    

    @staticmethod
    def p_label_blocked( label : str, facts : list[list] ):
        """ Return true if the facts imply that a named object is blocked """
        for fact in facts:
            if (fact[0] == "Blocked") and (fact[1] == label):
                return True
        return False
    

    @staticmethod
    def get_label_support( label : str, facts : list[list] ):
        """ Return true if the facts imply that a pose is occupied by an object """
        for fact in facts:
            if (fact[0] == "Supported") and (fact[1] == label):
                return fact[2]
        return None
        

    @staticmethod
    def p_pose_occupied( q : int, facts : list[list] ):
        """ Return true if the facts imply that a pose is occupied by an object """
        return (Solver.get_fact_by_pose( q, facts ) is not None)
    

    @staticmethod
    def p_label_supported( label : str, facts : list[list] ):
        """ Return true if the facts imply that a pose is occupied by an object """
        return (Solver.get_label_support( label, facts ) is not None)


    def ground_facts( self, objs : list[GraspObj], goal : list = None ):
        """ Set facts from the object list """
        rtnFcs = deque()
        for obj in objs:
            rtnFcs.append( ['GraspObj', obj.label, obj.pose] )
        if goal is not None:

            ## Match Goal Facts ##
            goalFcts = Solver.get_goal_facts( goal )
            for rFact in rtnFcs:
                dMin = 6e10
                pMin = None
                for gFact in goalFcts:
                    if (rFact[0] == gFact[0] == "GraspObj") and (rFact[1] == gFact[1]):
                        # d_ij = euclidean_distance_between_symbols( rFact[2], gFact[2] )
                        d_ij = abs( rFact[2] - gFact[2] )
                        if (d_ij <= env_var("_ACCEPT_POSN_ERR")) and (d_ij < dMin):
                            dMin = d_ij
                            pMin = gFact[2]
                if pMin is not None:
                    rFact[2] = pMin
            
            ## Stacked State ##
            for dnPose in self.ascendingPoses[:-1]:
                upPose = dnPose + 1
                dnFact = Solver.get_fact_by_pose( dnPose, rtnFcs )
                upFact = Solver.get_fact_by_pose( upPose, rtnFcs )
                if (upFact is not None) and (dnFact is not None):
                    rtnFcs.extend([
                        ["Blocked"  , dnFact[2], ],
                        ['Supported', upFact[1], dnFact[1], ]
                    ])

        ## Cleanup ##
        for label in _NAMES:
            if not Solver.p_label_supported( label, rtnFcs ):
                rtnFcs.append( ["Supported", label, "table",] )

        return list( rtnFcs )


    def p_fact_match( self, qFact, factList ):
        """ Return True if `qFact` is SUPPORTED by `factList` """
        for fact in factList:
            if (qFact[0] == fact[0]) and (qFact[1] == fact[1]):
                # if euclidean_distance_between_symbols( qFact[2], fact[2] ) <= env_var("_ACCEPT_POSN_ERR"):
                if abs( qFact[2] - fact[2] ) <= env_var("_ACCEPT_POSN_ERR"):
                    return True
        return False


    def p_wrong_object( self, qFact, factList ):
        """ Return True if `qFact` is CONTRADICTED by `factList` """
        for fact in factList:
            if (qFact[0] == fact[0]) and (qFact[1] != fact[1]):
                # if euclidean_distance_between_symbols( qFact[2], fact[2] ) <= env_var("_ACCEPT_POSN_ERR"):
                if abs( qFact[2] - fact[2] ) <= env_var("_ACCEPT_POSN_ERR"):
                    return True
        return False
    

    def get_object_pose( self, qLabel, factList ):
        """ Return the pose of the object matching `qLabel` in `factList`, else return `None` """
        for fact in factList:
            if (fact[0] == "GraspObj") and (fact[1] == qLabel):
                return fact[2]
        return None
    

    def p_fact_collide( self, qPose, facts ):
        """ Will the `q` collide with any of the current `facts` """
        for fact in facts:
            if (fact[0] == "GraspObj") and (euclidean_distance_between_symbols( qPose, fact[2] ) <= env_var("_ACCEPT_POSN_ERR")):
                return True
        return False


    def get_random_table_pose( self, facts : list[list], scale = 1.000 ):
        """ Get a table `ObjPose` that does not interfere with any of the current blocks """
        hlfScl = scale / 2.0

        def gen():
            """ Return a Random Pose """
            p = np.eye(4)
            x = -hlfScl + scale*random()
            y = -hlfScl + scale*random()
            p[0:3,3] = [x,y,env_var("_BLOCK_SCALE")/2.0,]
            return ObjPose(p)
        
        collide = True
        rtnPose = None
        while collide:
            rtnPose = gen()
            collide = self.p_fact_collide( rtnPose, facts )
        return rtnPose
    

    def p_goal_objects_present( self, obsL, goal ):
        """ Check that all the goal objects are present """
        gSet = set([g[1] for g in goal if g[0] == "GraspObj"])
        oSet = set([o[1] for o in obsL if o[0] == "GraspObj"])
        if len( oSet ) < len( gSet ):
            return False
        for g in gSet:
            if g not in oSet:
                return False
        return True


    def solve( self, facts : list[list], goal : list[list] ):
        """ Return a plan that solves the goal, If already solved then return an empty list """
        self.status = Status.RUNNING
        if not self.p_goal_objects_present( facts, goal ):
            self.status = Status.FAILURE
            return None
        goals  = self.get_goal_facts( goal )
        crrct  = list()
        replc  = list()
        empty  = list()
        gLen   = len( goals )
        height = 0
        for i, g in enumerate( goals ):
            if self.p_fact_match( g, facts ):
                crrct.append(i)
                height += 1
            elif self.p_wrong_object( g, facts ):
                replc.append(i)
                height += 1
            else:
                empty.append(i)
        cLen = len( crrct )
        rLen = len( replc )
        eLen = len( empty )
        plan = deque()
        if cLen >= gLen:
            print( f"SOLVED {cLen}: Return empty plan!" )
        if rLen > 0:
            print( f"INCORRECT: Need to undo {rLen} previous actions!" )
            for i in range( height-1, replc[0]-1, -1 ):
                # ASSUME: `get_random_table_pose()` WILL GENERALLY NOT CHOOSE POSES COLLIDING WITH PREVIOUS RUNS
                freePose = self.get_random_table_pose( facts )
                if i == 0:
                    plan.append( ["Place", facts[i][1], facts[i][2], freePose,] )
                else:
                    plan.append( ["Unstack", facts[i][1], facts[i][2], freePose,] )
        if eLen > 0:
            print( f"EMPTY: Need to build {eLen} of the tower!" )
            # for i in range( eLen ):
            for i in empty:
                if i == 0:
                    plan.append( ["Place", goals[i][1], facts[i][2]  ,  goals[i][2],] )
                else:
                    plan.append( ["Stack", goals[i][1], goals[i-1][1], facts[i][2], goals[i][2], goals[i-1][2],] )
        return list( plan )



########## SIMPLEST PLANNER ########################################################################
_MAX_ALLOWED_STEPS = 30

class FailModes( Enum ):
    """ Ways the planner can fail """
    PLANNING = "PLANNING FAILURE"
    ACTION   = "ACTION FAILURE"
    GOAL     = "GOAL OBJECTS NOT PRESENT"
    OKAY     = "STATUS OKAY - NO FAILURE"
    TIMEOUT  = "TIMEOUT FAILURE"


class SimExec:
    """ Manages the simulation """
    def __init__( self ):
        """ Set up the `Engine` and the `Solver` """
        self.obs    = list()
        self.facts  = list()
        self.engine = Engine()
        self.solver = Solver()
        self.Nstp   = 0
        self.status = Status.INVALID
        self.flMode = FailModes.OKAY


    def observe( self ):
        """ Simulate one run of the Perception Stack with possible confusion """
        self.obs : list[GraspObj] = self.engine .noisy_sense()
        print( "# Observed: #" )
        for ob in self.obs:
            print( f"\t{ob}" )


    def solve( self ):
        """ Get a plan given the current state """
        # self.plan = self.solver.solve( self.solver.ground_facts( self.engine .obss ), env_var("_GOAL_SIM") )
        self.facts = self.solver.ground_facts( self.obs )
        self.plan  = self.solver.solve( self.facts, env_var("_GOAL_SIM") )
        return self.plan


    def get_obs_by_label( self, lbl : str ):
        """ Get the current observation matching `lbl` """
        for ob in self.obs:
            print( f"\t\t{ob.label} -vs- {lbl}" )
            if ob.label == lbl:
                return ob
        return None


    def exec_step( self ):
        """ Execute the first action of the plan only """
        if len( self.plan ):
            action = self.plan[0]
            print( f"Execute: {action} at Step {self.Nstp}" )
            if action[0] == "Place":
                self.engine.place( action )
            elif action[0] == "Stack":
                self.engine.stack( action )
            elif action[0] == "Unstack":
                 self.engine.unstack( action )
            else:
                raise ValueError( f"CANNOT PARSE ACTION: {action}" )
        else:
            print( "NO PLAN TO EXECUTE" )


    def run_step( self ):
        """ Run entire cycle for one step of the plan """
        self.status = Status.RUNNING
        self.flMode = FailModes.OKAY
        self.Nstp  += 1
        # 1. Observe 
        self.observe()
        # 2. Plan 
        plan = self.solve()
        if plan is None:
            self.status = Status.FAILURE
            self.flMode = FailModes.GOAL
            return self.status
        if not len( plan ):
            self.status = Status.SUCCESS
            print( "GOAL ACHIEVED" )
        for action in plan:
            print( f"\t{action}" )
        # 3. Execute One Action 
        self.exec_step()
        return self.status
    

    def run_episode( self ):
        """ Run until solved """
        run = True 
        while run:
            self.run_step()
            if (self.plan is not None) and (not len( self.plan )):
                run = False
            if self.Nstp >= _MAX_ALLOWED_STEPS:
                self.flMode = FailModes.TIMEOUT
                run = False


    def report_status( self ):
        """ Print status and failure mode """
        print( f"{self.status.value}, {self.flMode.value}" )
    

########## MAIN ####################################################################################
if __name__ == "__main__":
    plnr = SimExec()

    for _ in range(5):
        plnr.run_step() 
    
    # plnr.run_episode()
    
    plnr.report_status()
    


########## EXIT ####################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 