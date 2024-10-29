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
from aspire.utils import ( DataLogger, )
from aspire.actions import ( BT_Runner, Interleaved_MoveFree_and_PerceiveScene, MoveFree, GroundedAction, )
from aspire.BlocksTask import set_blocks_env, BlockFunctions

### ASPIRE::PDDLStream ### 
from aspire.pddlstream.pddlstream.language.generator import from_gen_fn, from_test
from aspire.SymPlanner import SymPlanner

### Local ###
from obj_ID_server import Perception_OWLViT
from EROM import EROM, reify_chosen_beliefs
from draw_beliefs import render_memory_list



########## HELPER FUNCTIONS ########################################################################


def set_experiment_env():
    """ Params for this experiment """

    _poseGrn = np.eye(4)
    _poseGrn[0:3,3] = [ -0.091-0.070, # env_var("_MIN_X_OFFSET")+env_var("_X_WRK_SPAN")/2.0, 
                        -0.304, # env_var("_MIN_Y_OFFSET")+env_var("_Y_WRK_SPAN")/2.0, 
                         0.5*env_var("_BLOCK_SCALE"), ]
    _trgtGrn = ObjPose( _poseGrn )
    env_sto( "_DEF_NULL_SCORE"   , 0.75     )
    env_sto( "_NULL_EVIDENCE"    , True     )
    env_sto( "_UPDATE_PERIOD_S"  , 600.0    )
    env_sto( "_REIFY_SUPER_BEL"  ,   1.01   )
    
    env_sto( "_Z_SNAP_BOOST"     ,   0.00*env_var("_BLOCK_SCALE")   )
    env_sto( "_Z_STACK_BOOST"    ,   0.00*env_var("_BLOCK_SCALE")   )

    env_sto( "_USE_TIMEOUT"      , False    )
    env_sto( "_OBJ_TIMEOUT_S"    , 180.0    ) # Readings older than this are not considered
    env_sto( "_USE_DECAY"        , False    )
    env_sto( "_SCORE_DECAY_TAU_S", 120.0    )

    env_sto( "_N_INTAKE_SCANS"   ,   1     )

    env_sto( "_CUT_INTAKE_S_FRAC",   0.400  ) # 0.125 # 0.250 # 0.333 # 0.400 # 0.500 # 0.600  # 0.750
    env_sto( "_CUT_LKG_S_FRAC"   ,   0.250  ) # 0.125 # 0.333 # 0.400   
    env_sto( "_CUT_MERGE_S_FRAC" ,   0.250  ) # 0.125 # 0.333
    env_sto( "_CUT_DETERM_S_FRAC",   0.250  ) # 0.250 # 0.333

    env_sto( "_N_XTRA_SPOTS"     ,   3      )
    env_sto( "_N_REQD_OBJS"      ,   3      )
    env_sto( "_CONFUSE_PROB"     ,   0.025  )
    env_sto( "_SCORE_FILTER_EXP" ,   0.75   )
    env_sto( "_NULL_THRESH"      ,   0.625  ) # 0.50 # 0.75

    env_sto( "_SCORE_BIGNUM"      , 4000.00    )
    env_sto( "_SCORE_MULT_SUCCESS",    5.00    )
    env_sto( "_SCORE_MULT_DETERM" ,    1.25    )
    env_sto( "_N_MISS_PUNISH"     ,    2       )
    env_sto( "_WIPE_ON_FAILURE"   , True       )

    env_sto( "_LKG_SEP"          , 0.80*env_var("_BLOCK_SCALE") )  # 0.40 # 0.60 # 0.70 # 0.75
    env_sto( "_MAX_UPDATE_RAD_M" , 1.25*env_var("_BLOCK_SCALE") )
    env_sto( "_WIDE_XY_ACCEPT"   , 2.50*env_var("_BLOCK_SCALE") )
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

    env_sto( "_HACKED_OFFSET"  , np.array( [[ 1.0, 0.0, 0.0, -1.0/100.0, ],
                                            [ 0.0, 1.0, 0.0, -0.5/100.0, ],
                                            [ 0.0, 0.0, 1.0, 0.0, ],
                                            [ 0.0, 0.0, 0.0, 1.0, ],] ) )
    


    


########## PLANNER #################################################################################

class TaskPlanner:
    """ Basic task planning loop """

    ##### File Ops ########################################################

    def open_file( self ):
        """ Set the name of the current file """
        dateStr     = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.outNam = f"Task-Planner_{dateStr}.json"
        if (self.outFil is not None) and (not self.outFil.closed):
            self.outFil.close()
        self.outFil = open( os.path.join( self.outDir, self.outNam ), 'w' )


    def dump_to_file( self, openNext = False ):
        """ Write all data lines to a file """
        # self.outFil.writelines( [f"{str(line)}\n" for line in self.datLin] )        
        json.dump( self.datLin, self.outFil )
        self.outFil.close()
        self.datLin = list()
        if openNext:
            self.open_file()


    ##### Init ############################################################

    def reset_memory( self ):
        """ Erase belief memory """
        self.memory  = EROM() # Entropy-Ranked Object Memory


    def reset_state( self ):
        """ Erase problem state """
        self.status = Status.INVALID # Running status


    def __init__( self, noBot = False ):
        """ Create a pre-determined collection of poses and plan skeletons """
        set_blocks_env()
        set_experiment_env()
        self.datLin = list() # Data to write
        self.outFil = None
        self.noBot  = noBot
        self.outDir = "data/"
        self.reset_memory()
        self.reset_state()
        self.open_file()
        self.perc   = Perception_OWLViT
        self.robot : UR5_Interface = UR5_Interface() if (not noBot) else None
        self.logger = DataLogger() if (not noBot) else None
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
        self.dump_to_file( openNext = False )
        if not self.noBot:
            self.robot.reset_gripper_overload( restart = False )
            self.robot.stop()
            self.perc.shutdown()


    def p_failed( self ):
        """ Has the system encountered a failure? """
        return (self.status == Status.FAILURE)
    

    def return_home( self, goPose ):
        """ Get ready for next iteration while updating beliefs """
        btAction = GroundedAction( args = list(), robot = self.robot, name = "Return Home" )
        btAction.add_children([
            Open_Gripper( ctrl = self.robot ),
            MoveFree( [None, ObjPose( goPose )], robot = self.robot, suppressGrasp = True ), 
            # Interleaved_MoveFree_and_PerceiveScene( 
            #     MoveFree( [None, ObjPose( goPose )], robot = self.robot, suppressGrasp = True ), 
            #     self.symPln, 
            #     env_var("_UPDATE_PERIOD_S"), 
            #     initSenseStep = True 
            # ),
        ])
        
        btr = BT_Runner( btAction, env_var("_BT_UPDATE_HZ"), env_var("_BT_ACT_TIMEOUT_S") )
        btr.setup_BT_for_running()

        while not btr.p_ended():
            btr.tick_once()
            btr.per_sleep()

        print( f"\nRobot returned to \n{goPose}\n" )


    ##### Task Planning Phases ############################################

    def phase_1_Perceive( self, Nscans = 1 ):
        """ Take in evidence and form beliefs """

        camPose = self.robot.get_cam_pose()
        # camPose = camPose.dot( env_var("_HACKED_OFFSET") )

        # camPose = env_var("_HACKED_OFFSET").dot( self.robot.get_cam_pose() )
        
        obsrv = dict()
        for _ in range( Nscans ):
            obsrv.update( self.perc.build_model( shots = 1 ) )

        self.memory.process_observations( 
            obsrv,
            camPose 
        ) 

        # self.memory.get_current_most_likely()

        # reify_chosen_beliefs( self.memory.LKG , self.symPln.symbols, factor = env_var("_REIFY_SUPER_BEL") )


    def phase_2_Conditions( self ):
        """ Get the necessary initial state, Check for goals already met """
        self.symPln.symbols = self.memory.get_current_most_likely()
        reify_chosen_beliefs( self.symPln.symbols )

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
        self.symPln.set_update_funcs(
            self.phase_1_Perceive,
            self.p_belief_dist_OK
        )
        self.symPln.plan_task( 
            pdls_stream_map = {
                ### Symbol Streams ###
                'sample-above' : from_gen_fn( self.blcMod.get_above_pose_stream()     ), 
                ### Symbol Tests ###
                'test-free-placment': from_test( self.blcMod.get_free_placement_test() ),
            },
            robot    = self.robot
        )
        if (self.symPln.status == Status.FAILURE):
            self.status = Status.FAILURE
            self.logger.log_event( "Planning Failure" )
            print( f"Planning Failure!" )
        elif (self.symPln.status == Status.SUCCESS):
            self.status = Status.RUNNING
            print( f"\n\nPlanner thinks we SUCCEEDED!\n\n" )

        # reify_chosen_beliefs( self.memory.LKG , self.symPln.symbols, factor = env_var("_REIFY_SUPER_BEL") )


    def phase_4_Execute_Action( self ):
        """ Attempt to execute the first action in the symbolic plan """
        
        btr = BT_Runner( self.symPln.nxtAct, env_var("_BT_UPDATE_HZ"), env_var("_BT_ACT_TIMEOUT_S") )
        btr.setup_BT_for_running()

        lastTip = None
        currTip = None

        while not btr.p_ended():
            
            currTip = btr.tick_once()
            if currTip != lastTip:
                self.logger.log_event( f"Behavior: {currTip}", str(btr.status) )
            lastTip = currTip
            
            if (btr.status == Status.FAILURE):
                self.status = Status.FAILURE
                self.logger.log_event( "Action Failure", btr.msg )
            else:
                self.status = Status.RUNNING

            btr.per_sleep()

        self.logger.log_event( "BT END", str( btr.status ) )

        print( f"Did the BT move a reading?: {self.memory.move_reading_from_BT_plan( self.symPln.nxtAct )}" )


    def phase_5_Return_Home( self, goPose ):
        """ Get ready for next iteration while updating beliefs """
        self.return_home( goPose )
        

    ##### Task Planner Main Loop ##########################################

    def p_belief_dist_OK( self ): 
        """ Return False if belief change criterion met, Otherwise return True """
        print( f"\nFIXME: `ResponsiveTaskPlanner.p_belief_dist_OK` HAS NOT BEEN IMPLEMENTED!!!\n", file = sys.stderr )
        return True


    def solve_task( self, maxIter = 100, beginPlanPose = None ):
        """ Solve the goal """
        
        if beginPlanPose is None:
            if env_var("_BLOCK_SCALE") < 0.030:
                beginPlanPose = env_var("_GOOD_VIEW_POSE")
            else:
                beginPlanPose = env_var("_HIGH_VIEW_POSE")

        i = 0

        print( "\n\n\n##### TASK BEGIN #####\n" )

        # self.reset_beliefs() 
        self.reset_state() 
        
        self.symPln.set_goal( env_var("_GOAL") )

        self.logger.begin_trial()

        indicateSuccess = False
        t5              = now()

        

        while (self.status != Status.SUCCESS) and (i < maxIter): # and (not self.PANIC):

            
            self.status = Status.RUNNING

            print( f"### Iteration {i+1} ###" )
            
            i += 1

            ##### Phase 1 ########################

            print( f"Phase 1, {self.status} ..." )

            # self.set_goal()

            expBgn = now()
            # if (expBgn - t5) < env_var("_UPDATE_PERIOD_S"):
            #     sleep( env_var("_UPDATE_PERIOD_S") - (expBgn - t5) )

            self.robot.moveL( beginPlanPose, asynch = False ) # 2024-07-22: MUST WAIT FOR ROBOT TO MOVE            
            self.phase_1_Perceive( env_var("_N_INTAKE_SCANS") )

            # if 1:    
            #     self.robot.moveL( _SHOT_1, asynch = False ) # 2024-07-22: MUST WAIT FOR ROBOT TO MOVE            
            #     self.phase_1_Perceive( 1 )

            # if 0:
            #     self.robot.moveL( _SHOT_2, asynch = False ) # 2024-07-22: MUST WAIT FOR ROBOT TO MOVE            
            #     self.phase_1_Perceive( 1 )
            
            if self.status == Status.FAILURE:
                print( f"LOOP, {self.status} ..." )
                continue

            ##### Phase 2 ########################

            print( f"Phase 2, {self.status} ..." )
            self.phase_2_Conditions()

            if self.symPln.validate_goal_noisy( self.symPln.goal ):
                indicateSuccess = True
                self.logger.log_event( "Believe Success", f"Iteration {i}: Noisy facts indicate goal was met!\n{self.symPln.facts}" )
                print( f"!!! Noisy success at iteration {i} !!!" )
                self.status = Status.SUCCESS
            else:
                indicateSuccess = False

            if self.status in (Status.SUCCESS, Status.FAILURE):
                print( f"LOOP, {self.status} ..." )
                continue

            ##### Phase 3 ########################

            print( f"Phase 3, {self.status} ..." )
            self.phase_3_Plan_Task()

            if self.status in (Status.SUCCESS, Status.FAILURE):
                print( f"LOOP, {self.status} ..." )
                continue

            # DEATH MONITOR
            if self.symPln.noSoln >= self.symPln.nonLim:
                self.logger.log_event( "SOLVER BRAINDEATH", f"Iteration {i}: Solver has failed {self.symPln.noSoln} times in a row!" )
                break

            if self.p_failed():
                print( f"LOOP, {self.status} ..." )
                continue

            ##### Phase 4 ########################

            print( f"Phase 4, {self.status} ..." )

            if env_var("_USE_GRAPHICS"):
                render_memory_list( self.memory.LKG + self.memory.beliefs.beliefs, self.symPln.symbols )

            t4 = now()
            # if (t4 - expBgn) < env_var("_UPDATE_PERIOD_S"):
            #     sleep( env_var("_UPDATE_PERIOD_S") - (t4 - expBgn) )
            self.phase_4_Execute_Action()

            if self.p_failed():
                self.robot.open_gripper()
                if env_var("_WIPE_ON_FAILURE"):
                    self.memory.reset_memory()
                

            ##### Phase 5 ########################

            print( f"Phase 5, {self.status} ..." )
            t5 = now()
            # if (t5 - t4) < env_var("_UPDATE_PERIOD_S"):
            #     sleep( env_var("_UPDATE_PERIOD_S") - (t5 - t4) )
            self.phase_5_Return_Home( beginPlanPose )

            print()

        if 0: #self.PANIC:
            print( "\n\nWARNING: User-requested shutdown or other fault!\n\n" )
            self.logger.end_trial(
                False,
                {'PANIC': True, 'end_symbols' : list( self.symPln.symbols ), }
            )
        else:
            self.logger.end_trial(
                indicateSuccess,
                {'end_symbols' : list( self.symPln.symbols ) }
            )

        self.logger.save( "data/Baseline" )

        print( f"\n##### PLANNER END with status {self.status} after iteration {i} #####\n\n\n" )



########## EXPERIMENT HELPER FUNCTIONS #############################################################

_GOOD_VIEW_POSE = None
_HIGH_VIEW_POSE = None

def responsive_experiment_prep( beginPlanPose = None ):
    """ Init system and return a ref to the planner """
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
_VISION_TEST    = 0

_HIGH_TWO_POSE  = None


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


_EXP_BGN_POSE = _SHOT_4


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
            planner = responsive_experiment_prep( _EXP_BGN_POSE ) # _EXP_BGN_POSE
            planner.solve_task( maxIter = 30, beginPlanPose = _EXP_BGN_POSE )
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

    
        