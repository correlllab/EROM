"""
TaskPlanner.py
Correll Lab, CU Boulder
Contains the Baseline and Responsive Planners described in FIXME: INSERT PAPER REF AND DOI
Version 2024-07
Contacts: {james.watson-2@colorado.edu,}
"""
########## INIT ####################################################################################

##### Imports #####

### Standard ###
import sys, time, os, json
now = time.time
from time import sleep
from pprint import pprint
# from random import random
from traceback import print_exc, format_exc
from datetime import datetime
from math import isnan


### Special ###
import numpy as np
from py_trees.common import Status
# from py_trees.composites import Sequence
from magpie_control.BT import Open_Gripper
from magpie_control.ur5 import UR5_Interface
from magpie_control.poses import repair_pose
# import open3d as o3d

### ASPIRE ###
from aspire.env_config import env_var, env_sto
from aspire.symbols import ( ObjPose, )
from aspire.actions import ( BT_Runner, MoveFree, GroundedAction, )
from aspire.BlocksTask import set_blocks_env, BlockFunctions

### ASPIRE::PDDLStream ### 
from aspire.pddlstream.pddlstream.language.generator import from_gen_fn, from_test
from aspire.SymPlanner import SymPlanner

### Local ###
# from obj_ID_server import Perception_OWLViT
from OWLv2_Segment import Perception_OWLv2, _QUERIES

from Memory import Memory
from draw_beliefs import render_memory_list, render_scan_list, vispy_geo_list_window, table_geo




########## HELPER FUNCTIONS ########################################################################


def set_experiment_env():
    """ Params for this experiment """

     # 3D Printed Blocks

    _poseGrn = np.eye(4)
    _poseGrn[0:3,3] = [ -0.211, # env_var("_MIN_X_OFFSET")+env_var("_X_WRK_SPAN")/2.0, 
                        -0.463, # env_var("_MIN_Y_OFFSET")+env_var("_Y_WRK_SPAN")/2.0, 
                         0.5*env_var("_BLOCK_SCALE")+env_var("_Z_TABLE"), ]
    _trgtGrn = ObjPose( _poseGrn )
    
    env_sto( "_SCAN_ALPHA", 0.50 )

    env_sto( "_Z_SNAP_BOOST"     ,  -0.75*env_var("_BLOCK_SCALE")   )
    env_sto( "_Z_STACK_BOOST"    ,   0.00*env_var("_BLOCK_SCALE")   )

    env_sto( "_N_INTAKE_SCANS"   ,   1     )

    env_sto( "_N_XTRA_SPOTS"     ,   3      )
    env_sto( "_N_REQD_OBJS"      ,   3      )
    env_sto( "_CONFUSE_PROB"     ,   0.025  )

    env_sto( "_PLACE_XY_ACCEPT"  , 0.30*env_var("_BLOCK_SCALE") )
    env_sto( "_WIDE_XY_ACCEPT"   , 0.75*env_var("_BLOCK_SCALE") )

    env_sto( "_WIDE_Z_ABOVE"     , 1.75*env_var("_BLOCK_SCALE") )

    env_sto( "_ROBOT_FREE_SPEED",  0.125 ) 
    env_sto( "_ROBOT_HOLD_SPEED",  0.125 )
    env_sto( "_ACCEPT_POSN_ERR" ,  0.60*env_var( "_BLOCK_SCALE" ) ) # 0.75 # 0.90
    
    env_sto( "_GOAL" ,
        ( 'and',
            ('GraspObj', 'grnBlock' , _trgtGrn  ), # ; Tower
            ('Supported', 'ylwBlock', 'grnBlock'), 
            ('Supported', 'bluBlock', 'ylwBlock'), 
            ('HandEmpty',),
        )
    )

    env_sto( "_ANGRY_PUSH_M", 0.035 ) 
    


def basic_BT_run( btAction ):
    """ Run a basic BT with `BT_Runner` defaults """
    btr = BT_Runner( btAction, env_var("_BT_UPDATE_HZ"), env_var("_BT_ACT_TIMEOUT_S") )
    btr.setup_BT_for_running()

    while not btr.p_ended():
        btr.tick_once()
        btr.per_sleep()        



########## PLANNER #################################################################################

class TaskPlanner:
    """ Basic task planning loop """


    ##### Init ############################################################

    def reset_memory( self ):
        """ Erase belief memory """
        self.memory.reset_memory()


    def reset_state( self ):
        """ Erase problem state """
        self.status = Status.INVALID # Running status


    def __init__( self, noBot = False ):
        """ Create a pre-determined collection of poses and plan skeletons """
        set_blocks_env()
        set_experiment_env()
        self.outFil  = None
        self.noBot   = noBot
        self.status  = Status.INVALID # Running status

        # self.perc    = Perception_OWLViT
        self.perc = Perception_OWLv2()

        self.robot : UR5_Interface = UR5_Interface() if (not noBot) else None

        self.memory = Memory( self.robot ) 

        self.symPln = SymPlanner(
            os.path.join( os.path.dirname( __file__ ), "pddl", "domain.pddl" ),
            os.path.join( os.path.dirname( __file__ ), "pddl", "stream.pddl" )
        )
        self.blcMod = BlockFunctions( self.symPln )
        if (not noBot):
            self.robot.start()
            self.perc.start_vision()


    def shutdown( self ):
        """ Stop the Perception Process and the UR5 connection """
        self.memory.history.dump_to_file()
        if not self.noBot:
            self.robot.reset_gripper_overload( restart = False )
            self.robot.stop()
            self.perc.shutdown()


    def p_failed( self ):
        """ Has the system encountered a failure? """
        return (self.status == Status.FAILURE)
    

    def return_home( self, goPose ):
        """ Get ready for next iteration while updating beliefs """
        if isinstance( goPose, list ):
            goPose = goPose[0]
        btAction = GroundedAction( args = list(), robot = self.robot, name = "Return Home" )
        btAction.add_children([
            Open_Gripper( ctrl = self.robot ),
            MoveFree( [None, ObjPose( goPose )], robot = self.robot, suppressGrasp = True ), 
        ])
        basic_BT_run( btAction )
        print( f"\nRobot returned to \n{goPose}\n" )


    ##### Task Planning Phases ############################################

    def phase_1_Perceive( self, Append = False ):
        """ Take in evidence and form beliefs """

        camPose  = self.robot.get_cam_pose()

        obsrv, metadata = self.perc.segment( _QUERIES )

        self.memory.history.append( msg = "ObsMeta", datum = metadata )

        self.memory.process_observations( 
            obsrv,
            camPose,
            Append
        ) 


    def phase_2_Conditions( self ):
        """ Get the necessary initial state, Check for goals already met """
        self.symPln.symbols = self.memory.get_current_most_likely()

        if len( self.symPln.symbols ):
            self.status = Status.RUNNING
            if env_var("_VERBOSE"):
                print( f"\nStarting Objects:" )
                for obj in self.symPln.symbols:
                    print( f"\t{obj}" )
        else:
            self.status = Status.FAILURE
            if env_var("_VERBOSE"):
                print( f"\tNO OBJECTS DETERMINIZED" )

        self.blcMod.instantiate_conditions( self.robot )
        

    def phase_3_Plan_Task( self ):
        """ Attempt to solve the symbolic problem """
        self.symPln.plan_task( 
            pdls_stream_map = {
                ### Symbol Streams ###
                'sample-above' : from_gen_fn( self.blcMod.get_above_pose_stream()     ), 
                ### Symbol Tests ###
                'test-free-placment': from_test( self.blcMod.get_free_placement_test() ),
            },
            robot = self.robot
        )
        if (self.symPln.status == Status.FAILURE):
            self.status = Status.FAILURE
            self.memory.history.append( msg = "Planning Failure" )
            print( f"Planning Failure!" )
        elif (self.symPln.status == Status.SUCCESS):
            self.status = Status.RUNNING
            print( f"\n\nPlanner thinks we SUCCEEDED!\n\n" )


    def phase_4_Execute_Action( self ):
        """ Attempt to execute the first action in the symbolic plan """
        
        btr = BT_Runner( self.symPln.nxtAct, env_var("_BT_UPDATE_HZ"), env_var("_BT_ACT_TIMEOUT_S") )
        btr.setup_BT_for_running()

        lastTip = None
        currTip = None

        while not btr.p_ended():
            
            currTip = btr.tick_once()
            if currTip != lastTip:
                self.memory.history.append( msg = f"Behavior: {currTip}, {str(btr.status)}" )
            lastTip = currTip
            
            if (btr.status == Status.FAILURE):
                self.status = Status.FAILURE
                self.memory.history.append( msg = f"Action Failure: {btr.msg}" )
            else:
                self.status = Status.RUNNING

            btr.per_sleep()

        self.memory.history.append( msg = f"BT END: {btr.status}" )


    def phase_5_Return_Home( self, goPose ):
        """ Get ready for next iteration while updating beliefs """
        self.return_home( goPose )
        

    ##### Task Planner Main Loop ##########################################

    def p_belief_dist_OK( self ): 
        """ Return False if belief change criterion met, Otherwise return True """
        print( f"\nFIXME: `ResponsiveTaskPlanner.p_belief_dist_OK` HAS NOT BEEN IMPLEMENTED!!!\n", file = sys.stderr )
        return True


    def solve_task( self, maxIter, beginPlanPose ):
        """ Solve the goal """
        if not isinstance( beginPlanPose, list ):
            beginPlanPose = [beginPlanPose,]

        i = 0

        print( "\n\n\n##### TASK BEGIN #####\n" )

        self.reset_state() 
        
        self.symPln.set_goal( env_var("_GOAL") )

        self.memory.history.append( msg = "Task Start" )

        while (self.status != Status.SUCCESS) and (i < maxIter): # and (not self.PANIC):
            
            self.status = Status.RUNNING

            print( f"### Iteration {i+1} ###" )
            
            i += 1

            ##### Phase 1 ########################

            print( f"Phase 1, {self.status} ..." )

            # for bgnPose in beginPlanPose:

            bgnPoses = self.memory.plan_3d_shots( beginPlanPose[0] )
            self.memory.reset_memory()

            vispy_geo_list_window( [table_geo(),], robotPose = bgnPoses )

            for bgnPose in bgnPoses:
                self.robot.moveL( bgnPose, asynch = False ) # 2024-07-22: MUST WAIT FOR ROBOT TO MOVE            
                self.phase_1_Perceive( Append = True )

            if env_var("_USE_GRAPHICS"):
                render_scan_list( self.memory.scan )

            ##### Phase 2 ########################

            print( f"Phase 2, {self.status} ..." )
            self.phase_2_Conditions()

            if self.symPln.validate_goal_noisy( self.symPln.goal ):
                self.memory.history.append( msg = f"Believe Success, Iteration {i}: Noisy facts indicate goal was met!\n{self.symPln.facts}" )
                print( f"!!! Noisy success at iteration {i} !!!" )
                self.status = Status.SUCCESS

            if self.status in (Status.SUCCESS, Status.FAILURE):
                print( f"LOOP, {self.status} ..." )
                continue

            ##### Phase 3 ########################

            print( f"Phase 3, {self.status} ..." )
            self.phase_3_Plan_Task()

            if self.status in (Status.SUCCESS, Status.FAILURE):
                print( f"LOOP, {self.status} ..." )
                continue

            if self.p_failed():
                print( f"LOOP, {self.status} ..." )
                continue

            ##### Phase 4 ########################

            print( f"Phase 4, {self.status} ..." )

            if env_var("_USE_GRAPHICS"):
                render_memory_list( syms = self.symPln.symbols )

            self.phase_4_Execute_Action()

            if self.p_failed():
                self.robot.open_gripper()
                

            ##### Phase 5 ########################

            print( f"Phase 5, {self.status} ..." )
            self.phase_5_Return_Home( beginPlanPose )

            print()

        self.memory.history.append( 
            msg   = f"Task End, Succes?: {self.status}, end_symbols : {list( self.symPln.symbols )}",
            datum = list( self.symPln.symbols )
        )

        print( f"\n##### PLANNER END with status {self.status} after iteration {i} #####\n\n\n" )



########## EXPERIMENT HELPER FUNCTIONS #############################################################

_GOOD_VIEW_POSE = None
_HIGH_VIEW_POSE = None

def experiment_prep( beginPlanPose = None ):
    """ Init system and return a ref to the planner """

    if isinstance( beginPlanPose, list ):
        beginPlanPose = beginPlanPose[0]

    planner = TaskPlanner()
    planner.robot.set_grip_N( 10.0 )
    print( planner.robot.get_tcp_pose() )

    if beginPlanPose is None:
        if env_var("_BLOCK_SCALE") < 0.030:
            beginPlanPose = _GOOD_VIEW_POSE
        else:
            beginPlanPose = _HIGH_VIEW_POSE
    
    planner.robot.open_gripper()
    return planner

    

########## MAIN ####################################################################################

_TROUBLESHOOT   = 0



_CONF_CAM_POSE_ANGLED1 = repair_pose( np.array( [[ 0.55 , -0.479,  0.684, -0.45 ],
                                                 [-0.297, -0.878, -0.376, -0.138],
                                                 [ 0.781,  0.003, -0.625,  0.206],
                                                 [ 0.   ,  0.   ,  0.   ,  1.   ],] ) )

_YCB_LANDSCAPE_CLOSE_BGN = repair_pose( np.array( [[-0.698,  0.378,  0.608, -0.52 ],
                                                   [ 0.264,  0.926, -0.272, -0.308],
                                                   [-0.666, -0.029, -0.746,  0.262],
                                                   [ 0.   ,  0.   ,  0.   ,  1.   ],] ) )

_YCB_LANDSCAPE_FAR_BGN = repair_pose( np.array( [[-0.873,  0.238,  0.426, -0.474],
                                                 [ 0.206,  0.971, -0.121, -0.212],
                                                 [-0.442, -0.018, -0.897,  0.394],
                                                 [ 0.   ,  0.   ,  0.   ,  1.   ],] ) )


_SHOT_1 = repair_pose( np.array( [[-0.635,  0.251,  0.731, -0.615,],
                                  [ 0.172,  0.968, -0.182, -0.18 ,],
                                  [-0.753,  0.011, -0.658,  0.302,],
                                  [ 0.   ,  0.   ,  0.   ,  1.   ,],] ) )


_SHOT_3 = repair_pose( np.array( [[-0.824,  0.078,  0.562, -0.498,],
                                  [ 0.1  ,  0.995,  0.008, -0.26 ,],
                                  [-0.558,  0.063, -0.827,  0.379,],
                                  [ 0.   ,  0.   ,  0.   ,  1.   ,],] ) )


_SHOT_2 = repair_pose( np.array( [[-0.905,  0.17 ,  0.391, -0.44 ,],
                                  [ 0.116,  0.981, -0.158, -0.181,],
                                  [-0.41 , -0.098, -0.907,  0.513,],
                                  [ 0.   ,  0.   ,  0.   ,  1.   ,],] ) )


_SHOT_4 = repair_pose( np.array( [[-0.843,  0.018,  0.538, -0.476,],
                                  [ 0.056,  0.997,  0.054, -0.279,],
                                  [-0.535,  0.075, -0.841,  0.338,],
                                  [ 0.   ,  0.   ,  0.   ,  1.   ,],] ) )


_SHOT_5 = repair_pose( np.array( [[-0.705, -0.694,  0.144, -0.365],
                                  [-0.708,  0.678, -0.197, -0.322],
                                  [ 0.039, -0.24 , -0.97 ,  0.439],
                                  [ 0.   ,  0.   ,  0.   ,  1.   ],] ))


_SHOT_6 = repair_pose( np.array( [[-0.07,  -0.951, -0.3 ,  -0.059],
                                 [-0.995,  0.086 ,-0.04 , -0.38 ],
                                 [ 0.064,  0.296 ,-0.953,  0.457],
                                 [ 0.   ,  0.    , 0.   ,  1.   ],] ))
 

# _EXP_BGN_POSES = [_SHOT_6, _SHOT_6]
_EXP_BGN_POSES = [_SHOT_6,]


if __name__ == "__main__":

    dateStr = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    if _TROUBLESHOOT:
        print( f"########## Running Debug Code at {dateStr} ##########" )
        from aspire.homog_utils import R_x, homog_xform

        if 0:
            planner = TaskPlanner( noViz = True, noBot = True )
            blcPosn = {
                "good": [ 0.0  ,  0.0  ,  0.140,],
                "bad1": [ 0.0  ,  0.140,  0.0  ,],
                "bad2": [ 0.140,  0.0  ,  0.0  ,],
                "bad3": [ 0.0  , -0.140,  0.0  ,],
                "bad4": [-0.140,  0.0  ,  0.0  ,],
                "bad5": [ 0.0  ,  0.0  , -0.140,],
            }
            blcPose = np.eye(4)
            camPose = np.eye(4)
            camPose = camPose.dot( homog_xform( R_x(np.pi/2.0), [0,0,0] ) )

            for k, v in blcPosn.items():
                blcPose[0:3,3] = v
                print( f"Pose: {k}, Passed?: {planner.memory.p_symbol_in_cam_view( camPose, blcPose )}\n" )

        
        elif 1:
            rbt = UR5_Interface()
            rbt.start()
            sleep(2)
            print( f"Began at pose:\n{rbt.get_tcp_pose()}" )            
            rbt.stop()

    else:
        print( f"########## Running Planner at {dateStr} ##########" )

        try:
            planner = experiment_prep( _EXP_BGN_POSES ) # _EXP_BGN_POSE
            planner.solve_task( maxIter = 30, beginPlanPose = _EXP_BGN_POSES )
            sleep( 2.5 )
            planner.shutdown()
            

        except KeyboardInterrupt:
            # User Panic: Attempt to shut down gracefully
            print( f"\nSystem SHUTDOWN initiated by user!, Planner Status: {planner.status}\n" )
            print_exc()
            print()
            planner.shutdown()

        except Exception as e:
            # Bad Thing: Attempt to shut down gracefully
            print( f"Something BAD happened!: {e}" )
            print_exc()
            print()
            planner.shutdown()

    os.system( 'kill %d' % os.getpid() ) 

    
        