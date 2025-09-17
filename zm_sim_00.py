### Standard ### 
import os
from collections import deque
from random import random

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

class Engine:
    """ Shit Happens """
    def __init__( self ):
        self.objs = KNOWN_BLOCKS()
        self.prob = {
            "ActionFailure" : 0.10,
        }


    def report( self ):
        """ Print what is happening with the objects """
        print( "\n##### Current State of World Objects #####" )
        for obj in self.objs:
            print( obj )
        print()


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
        # self.goal = env_var("_GOAL_SIM")


    def get_goal_facts( self, goal : tuple ):
        """ Get the individual facts from the `goal` """
        if goal[0] in ("and", "or"):
            rtnFcs = list()
            for item in goal[1:]:
                rtnFcs.extend( self.get_goal_facts( item ) )
            return rtnFcs
        elif goal[0] == "not":
            return list()
        else:
            return [goal,]


    def ground_facts( self, objs : list[GraspObj], goal : tuple = None ):
        """ Set facts from the object list """
        rtnFcs = deque()
        for obj in objs:
            rtnFcs.append( ['GraspObj', obj.label, obj.pose.copy()] )
        if goal is not None:
            goalFcts = self.get_goal_facts( goal )
            for rFact in rtnFcs:
                dMin = 6e10
                pMin = None
                for gFact in goalFcts:
                    if (rFact[0] == gFact[0] == "GraspObj") and (rFact[1] == gFact[1]):
                        d_ij = euclidean_distance_between_symbols( rFact[2], gFact[2] )
                        if (d_ij <= env_var("_ACCEPT_POSN_ERR")) and (d_ij < dMin):
                            dMin = d_ij
                            pMin = gFact[2]
                if pMin is not None:
                    rFact[2] = pMin
        return [tuple( item ) for item in rtnFcs]


    def order_by_Z( self, facts : list[tuple] ):
        """ Return the object pose facts in increasing Z order """
        rtnFcs = [item for item in facts if (item[0] == "GraspObj")]
        rtnFcs.sort( key = lambda x: extract_pose_as_homog( x[2] )[2,3]  )
        return rtnFcs


    def p_fact_match( self, qFact, factList ):
        """ Return True if `qFact` is SUPPORTED by `factList` """
        for fact in factList:
            if (qFact[0] == fact[0]) and (qFact[1] == fact[1]):
                if euclidean_distance_between_symbols( qFact[2], fact[2] ) <= env_var("_ACCEPT_POSN_ERR"):
                    return True
        return False


    def p_wrong_object( self, qFact, factList ):
        """ Return True if `qFact` is CONTRADICTED by `factList` """
        for fact in factList:
            if (qFact[0] == fact[0]) and (qFact[1] != fact[1]):
                if euclidean_distance_between_symbols( qFact[2], fact[2] ) <= env_var("_ACCEPT_POSN_ERR"):
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


    def get_random_table_pose( self, facts : list[tuple], scale = 1.000 ):
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
    

    def solve( self, facts : list[tuple], goal : list[tuple] ):
        """ Return a plan that solves the goal, If already solved then return an empty list """
        facts  = self.order_by_Z( facts )
        goals  = self.order_by_Z( self.get_goal_facts( goal ) )
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
                    plan.append( ("Place", facts[i][1], freePose,) )
                else:
                    plan.append( ("Unstack", facts[i][1], freePose,) )
        if eLen > 0:
            print( f"EMPTY: Need to build {eLen} of the tower!" )
            for i in range( eLen ):
                if i == 0:
                    plan.append( ("Place", goals[i][1], goals[i][2],) )
                else:
                    plan.append( ("Stack", goals[i][1], goals[i-1][1], goals[i][2],) )
        return list( plan )

        


class SimPlanner:
    """ Dice Roll Planner, w/o PDLS """
    def __init__( self ):
        self.goal = None
        self.plan = None


########## MAIN ####################################################################################
if __name__ == "__main__":
    eng = Engine()
    eng.report()
    slv = Solver()
    pln = slv.solve( slv.ground_facts( eng.objs ), env_var("_GOAL_SIM") )

    print( "\n##### Plan #####" )
    for action in pln:
        print( action )
    


########## EXIT ####################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 