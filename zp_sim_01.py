### Standard ### 
import os, json
from collections import deque
from random import random, choice
from enum import Enum
from pprint import pprint
from copy import deepcopy

### Special ### 
import numpy as np

### ASPIRE::PDDLStream ### 
from aspire.symbols import ObjPose, GraspObj, euclidean_distance_between_symbols
from aspire.env_config import env_var, env_sto
from aspire.BlocksTask import set_blocks_env

### Local ### 
from env_config import set_experiment_env
# from env_config import KNOWN_BLOCKS



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


def copy_observations( obsLst : list[SimBlock] ):
    """ Deep copy of blocks """
    rtnLst = deque()
    for obs in obsLst:
        rtnLst.append( obs.copy() )
    return list( rtnLst )

########## EXPERIMENTAL DATA #######################################################################
"""
### Makespan Distribution [Steps], RGB ###
	KC-KP
	Mean: ___ 3.0
	Median: _ 3.0
	Std.Dev.: 0.0
	SC-KP
	Mean: ___ 9.666666666666666
	Median: _ 8.0
	Std.Dev.: 6.882229243862731
	KC-SP
	Mean: ___ 11.0
	Median: _ 7.0
	Std.Dev.: 7.419746058925964
	SC-SP
	Mean: ___ 11.45
	Median: _ 9.0
	Std.Dev.: 8.570151690606183

### Makespan Distribution [Time], RGB ###
	KC-KP
	Mean: ___ 451.8967197418213
	Median: _ 446.1805534362793
	Std.Dev.: 56.51808618172577
	SC-KP
	Mean: ___ 1090.7812630534172
	Median: _ 873.0658189058304
	Std.Dev.: 677.1761763595428
	KC-SP
	Mean: ___ 1378.7196902224891
	Median: _ 988.5193696022034
	Std.Dev.: 1117.6213611473825
	SC-SP
	Mean: ___ 1117.5333013032612
	Median: _ 982.4180154800415
	Std.Dev.: 677.8997286287095

### Makespan Distribution [Steps], RBW ###
	KC-KP
	Mean: ___ 3.1363636363636362
	Median: _ 3.0
	Std.Dev.: 0.6248966856757964
	SC-KP
	Mean: ___ 9.555555555555555
	Median: _ 7.0
	Std.Dev.: 6.693464658590681
	KC-SP
	Mean: ___ 6.6
	Median: _ 5.0
	Std.Dev.: 4.768647607026546
	SC-SP
	Mean: ___ 10.65
	Median: _ 8.0
	Std.Dev.: 7.491828882188915

### Makespan Distribution [Time], RBW ###
	KC-KP
	Mean: ___ 391.08244509924026
	Median: _ 377.42189955711365
	Std.Dev.: 42.62549141327943
	SC-KP
	Mean: ___ 1772.619768301646
	Median: _ 1249.6555206775665
	Std.Dev.: 1268.537725692636
	KC-SP
	Mean: ___ 1080.037369602605
	Median: _ 933.0967352390289
	Std.Dev.: 673.8784369834951
	SC-SP
	Mean: ___ 1837.397778570652
	Median: _ 1427.0049859285355
	Std.Dev.: 1509.0870768442205
"""


########## HELPER FUNCTIONS ########################################################################
_STATS_PATH = "data/allData.txt"
_STATS_DICT = dict()
with open( _STATS_PATH, 'r' ) as f:
    _STATS_DICT = json.load( f )
_VERBOSE = False
_NAMES   = ["A","B","C",]
_EXP_PROBS_TIMES = {
    "RGB": {
        "KC-KP" : {  
            "ActionFailure" :  0.00,  
            "planFailure"   :  0.00,  
            "searchFailure" :  0.00,  
            "classConfuse"  :  0.00,  
            "t_search"      : 97.85,  
            "t_action"      : 18.72,  
        },
        "SC-KP" : {  
            "ActionFailure" :  0.1060,  
            "planFailure"   :  0.2002,  
            "searchFailure" :  0.0508,  
            "classConfuse"  :  0.0573,  
            "t_search"      : 99.55,  
            "t_action"      : 13.73,  
        },
        "KC-SP" : {  
            "ActionFailure" :  0.1439,  
            "planFailure"   :  0.2589,  
            "searchFailure" :  0.0464,  
            "classConfuse"  :  0.0483,  
            "t_search"      : 92.71,  
            "t_action"      : 14.75,  
        },
        "SC-SP" : {  
            "ActionFailure" :  0.2549,  
            "planFailure"   :  0.1617,  
            "searchFailure" :  0.0035,  
            "classConfuse"  :  0.0598,  
            "t_search"      : 95.81,  
            "t_action"      : 12.38,  
        },
    }, 
    "RBW": {
        "KC-KP" : {  
            "ActionFailure" :  0.0152,  
            "planFailure"   :  0.0076,  
            "searchFailure" :  0.0032,  
            "classConfuse"  :  0.00,  
            "t_search"      : 83.43,  
            "t_action"      : 21.01,  
        },
        "SC-KP" : {  
            "ActionFailure" :   0.1614,  
            "planFailure"   :   0.2396,  
            "searchFailure" :   0.0670,  
            "classConfuse"  :   0.0589,  
            "t_search"      : 136.24,  
            "t_action"      :  14.36,  
        },
        "KC-SP" : {  
            "ActionFailure" :   0.0991,  
            "planFailure"   :   0.2774,  
            "searchFailure" :   0.1309,  
            "classConfuse"  :   0.0932,  
            "t_search"      : 125.68,  
            "t_action"      :  14.54,  
        },
        "SC-SP" : {  
            "ActionFailure" :   0.1692,  
            "planFailure"   :   0.2764,  
            "searchFailure" :   0.0744,  
            "classConfuse"  :   0.0464,  
            "t_search"      : 134.21,  
            "t_action"      :  13.66,  
        },
    }
}


########## TRANSITION MODEL ########################################################################

class Engine:
    """ Shit Happens """

    ##### Static Methods ##################################################

    ## Class Vars ##
    _poses  = set([i for i in range(3)])
    _bignum = 100000 # ASSUMPTION: WE DO NOT NEED MORE THAN `_bignum` POSES!


    @classmethod
    def reset_poses( cls ):
        cls._poses = set([i for i in range(3)])


    @staticmethod
    def roll_pose() -> int:
        return int( 3 + random() * Engine._bignum )


    @staticmethod
    def rand_pose() -> int:
        """ Generate a random int above 2 """
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

    def __init__( self, params_ = None ):
        """ Setup a New Episode """
        self.reset_poses()
        if params_ is None:
            self.params = {  
                "ActionFailure" : 0.10,  
                "classConfuse"  : 0.10,  
            }
        else:
            self.params = deepcopy( params_ )
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
            if random() >= self.params["ActionFailure"]:
                return True
            else:
                if _VERBOSE: 
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
            if (not forceFail) and (random() >= self.params["ActionFailure"]):
                return True
            else:
                if _VERBOSE: 
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
            if (not forceFail) and (random() >= self.params["ActionFailure"]):
                return True
            else:
                if _VERBOSE: 
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



class StatsRoller:
    """ Roll a value from the experimental data """
    def __init__( self, problem : str, scenario : str ):
        self.prblm = problem
        self.scnro = scenario

    def roll_from( self, param : str ):
        """ Choose from experimental data """
        return choice( _STATS_DICT[ self.prblm ][ self.scnro ][ param ] )



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
            if _VERBOSE: 
                print( f"SOLVED {cLen}: Return empty plan!" )
        if rLen > 0:
            if _VERBOSE: 
                print( f"INCORRECT: Need to undo {rLen} previous actions!" )
            for i in range( height-1, replc[0]-1, -1 ):
                freePose = Engine.rand_pose()
                if i == 0:
                    plan.append( ["Place", facts[i][1], facts[i][2], freePose,] )
                else:
                    plan.append( ["Unstack", facts[i][1], facts[i][2], freePose,] )
        if eLen > 0:
            if _VERBOSE: 
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
    def __init__( self, engParams : dict = None, roller : StatsRoller = None ):
        """ Set up the `Engine` and the `Solver` """
        self.obs    = list()
        self.oHist  = deque()
        self.facts  = list()
        self.engine = Engine( engParams )
        self.solver = Solver()
        self.Nstp   = 0
        self.status = Status.INVALID
        self.flMode = FailModes.OKAY
        self.roller = roller
        # Per-Episode Statistics
        self.result = { 
            "tRun"    : 0,
            "conf"    : deque(),
            "actnFail": 0,
            "planFail": 0,
            "Nsteps"  : 0,
            "success" : False,
        }


    def n_changed_labels( self ):
        """ Count the number of labels that changed between the last two observations """
        rtN = 0
        if len( self.oHist ) >= 2:
            last : list[SimBlock] = self.oHist[-1] 
            prev : list[SimBlock] = self.oHist[-2]
            for obj_l in last:
                conf = False
                for obj_p in prev:
                    if (abs(obj_l.pose - obj_p.pose) <= env_var("_ACCEPT_POSN_ERR")) and (obj_l.label != obj_p.label):
                        conf = True
                if conf:
                    rtN += 1
        return rtN


    def observe( self ):
        """ Simulate one run of the Perception Stack with possible confusion """
        self.obs : list[GraspObj] = self.engine.noisy_sense()
        self.oHist.append( copy_observations( self.obs ) )
        self.result["conf"].append( self.n_changed_labels() )

        # self.result["tRun"] += self.engine.params["t_search"]
        if self.roller is not None:
            self.result["tRun"] += self.roller.roll_from( "tObsTot" )
        else:
            # ASSUMPTION: SEARCH TIME KIND OF LOOKS LIKE A POISSON DISTRIBUTION
            self.result["tRun"] += np.random.poisson( self.engine.params["t_search"] )

        if _VERBOSE: 
            print( "# Observed: #" )
            for ob in self.obs:
                print( f"\t{ob}" )


    def solve( self ):
        """ Get a plan given the current state """
        # self.plan = self.solver.solve( self.solver.ground_facts( self.engine .obss ), env_var("_GOAL_SIM") )
        if random() < (self.engine.params["planFailure"] + self.engine.params["searchFailure"]):
            self.plan = None
        else:    
            self.facts = self.solver.ground_facts( self.obs )
            self.plan  = self.solver.solve( self.facts, env_var("_GOAL_SIM") )
        if self.plan is None:
            self.result["planFail"] += 1
        return self.plan


    def exec_step( self ):
        """ Execute the first action of the plan only """
        if len( self.plan ):
            action = self.plan[0]
            result = False
            self.result["tRun"] += self.engine.params["t_action"]
            if _VERBOSE: 
                print( f"Execute: {action} at Step {self.Nstp}" )
            if action[0] == "Place":
                result = self.engine.place( action )
            elif action[0] == "Stack":
                result = self.engine.stack( action )
            elif action[0] == "Unstack":
                result = self.engine.unstack( action )
            else:
                raise ValueError( f"CANNOT PARSE ACTION: {action}" )
            if not result:
                self.result["actnFail"] += 1
        else:
            if _VERBOSE: 
                print( "NO PLAN TO EXECUTE" )


    def run_step( self ):
        """ Run entire cycle for one step of the plan """
        self.status = Status.RUNNING
        self.flMode = FailModes.OKAY
        self.result["Nsteps"] += 1
        self.Nstp += 1
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
            if _VERBOSE: 
                print( "GOAL ACHIEVED" )
            return self.status
        if _VERBOSE: 
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
        if self.status == Status.SUCCESS:
            self.result["success"] = True
        if self.result["success"] == True:
            self.result["Nsteps"] -= 1
        return self.result


    def report_status( self ):
        """ Print status and failure mode """
        print( f"{self.status.value}, {self.flMode.value}" )
    

########## MAIN ####################################################################################
_N_EPISODES = 1000 # 100 # 1000 # 10000
_DIV_STATUS =   int( _N_EPISODES / 10 )
if __name__ == "__main__":
    for problem in _EXP_PROBS_TIMES.keys():
        for scenario in _EXP_PROBS_TIMES[ problem ].keys():
            stats = {
                "tMS" : 0.0,
                "sMS" : 0  ,
                "rSC" : 0  ,
                "rFL" : 0  ,
            }
            for i in range( _N_EPISODES ):
                plnr = SimExec( 
                    _EXP_PROBS_TIMES[ problem ][ scenario ],
                    StatsRoller( problem, scenario ) 
                )
                res_i = plnr.run_episode()
                if i % _DIV_STATUS == 0:
                    print('.',end='',flush=True)
                if _VERBOSE: 
                    pprint( res_i )
                    plnr.report_status()
                stats["tMS"] += res_i["tRun"  ]
                stats["sMS"] += res_i["Nsteps"]
                if res_i["success"]:
                    stats["rSC"] += 1
                else:
                    stats["rFL"] += 1
            print( f"\n##### {problem}::{scenario} Statistics ##### " )
            print( f"\t Makespan [s]: _ {stats['tMS']/_N_EPISODES:.2f}" )
            print( f"\t Makespan Steps: {stats['sMS']/_N_EPISODES:.2f}" )
            print( f"\t Success Rate: _ {stats['rSC']/_N_EPISODES:.2f}" )
            print()

    


########## EXIT ####################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 