"""
TaskPlanner.py
Correll Lab, CU Boulder
Contains the Baseline and Responsive Planners described in FIXME: INSERT PAPER REF AND DOI
Version 2024-07
Contacts: {james.watson-2@colorado.edu,}
"""
########## INIT ####################################################################################

##### Imports #############################################################

### Standard ###
import time, os
now = time.time
from time import sleep
# from random import random
from traceback import print_exc
from datetime import datetime


### Special ###
import numpy as np
from py_trees.common import Status
from magpie_control.BT import Open_Gripper, BT_Runner
from magpie_control.ur5 import UR5_Interface
from magpie_control.poses import repair_pose
from magpie_control.utils import vec_unit

### ASPIRE ###
from aspire.env_config import env_var, env_sto
from aspire.symbols import ( ObjPose, GraspObj, extract_pose_as_homog )
from aspire.BlocksTask import set_blocks_env, BlockFunctions
from aspire.actions.pdls_behaviors import GroundedAction, MoveFree, Plan
from aspire.actions.utils import line_intersect_plane

### ASPIRE::PDDLStream ### 
from aspire.pddlstream.pddlstream.language.generator import from_gen_fn, from_test
from aspire.SymPlanner import SymPlanner

### Local ###
from BT import ReactivePlanParser
from OWLv2_Segment import Perception_OWLv2, _QUERIES

from Memory import Memory, PoseCheater
from draw_beliefs import render_memory_list, render_scan_list


##### Globals #############################################################

_SAFE = repair_pose( np.array( [[-0.985, -0.163, -0.052, -0.252],
                                [-0.164,  0.986,  0.013, -0.262],
                                [ 0.049,  0.021, -0.999,  0.471],
                                [ 0.   ,  0.   ,  0.   ,  1.   ],] ) )

_BLOCK_TYPE = "plastic" if (env_var("_BLOCK_SCALE") > 0.030) else "wooden"
_BLOCK_DESC = {
    'grnBlock' : f"green {_BLOCK_TYPE} block",
    # 'ylwBlock' : f"yellow {_BLOCK_TYPE} block",
    'redBlock' : f"red {_BLOCK_TYPE} block",
    'bluBlock' : f"blue {_BLOCK_TYPE} block",
    'table'    : f"wooden table",
}

_RESPONSIVE_MODE = False

########## HELPER FUNCTIONS ########################################################################


def BASE_TARGET():
    _poseGrn = np.eye(4)
    _poseGrn[0:3,3] = [ -0.200, # env_var("_MIN_X_OFFSET")+env_var("_X_WRK_SPAN")/2.0, 
                        -0.300, # env_var("_MIN_Y_OFFSET")+env_var("_Y_WRK_SPAN")/2.0, 
                         0.5*env_var("_BLOCK_SCALE")+env_var("_Z_TABLE"), ]
    return ObjPose( _poseGrn )


def set_experiment_env():
    """ Params for this experiment """

    env_sto( "_OWL2_THRESH"  , 0.0025 ) # 0.005
    env_sto( "_SEG_MAX_HITS"    , 50     ) 
    env_sto( "_SEG_SCORE_THRESH",  0.100 ) # 0.025 # 0.075 # 0.100

    env_sto( "_Z_SAFE", 0.350 )

     # 3D Printed Blocks

    _trgtGrn = BASE_TARGET()
    
    env_sto( "_BLOCK_VOLUME", env_var( "_BLOCK_SCALE" )**3 )

    env_sto( "_VERBOSE"     , True )
    env_sto( "_USE_GRAPHICS", False )
    env_sto( "_SCAN_ALPHA"  , 0.35  )

    # env_sto( "_Z_SNAP_BOOST" , -0.25*env_var("_BLOCK_SCALE")   )
    # env_sto( "_Z_SNAP_BOOST" , 0.00*env_var("_BLOCK_SCALE") )
    env_sto( "_Z_SNAP_BOOST" , 0.125*env_var("_BLOCK_SCALE") )
    # env_sto( "_Z_SNAP_BOOST" , 0.25*env_var("_BLOCK_SCALE") )

    env_sto( "_Z_STACK_BOOST", 0.00*env_var("_BLOCK_SCALE") )
    # env_sto( "_Z_STACK_BOOST", 0.125*env_var("_BLOCK_SCALE") )

    env_sto( "_N_INTAKE_SCANS"   ,   1     )

    env_sto( "_N_XTRA_SPOTS",   3     )
    env_sto( "_N_REQD_OBJS" ,   3     )
    env_sto( "_CONFUSE_PROB",   0.025 )

    # env_sto( "_BAYES_RAD_L2_M" , 1.000*env_var("_BLOCK_SCALE")  )
    # env_sto( "_BAYES_RAD_L2_M" , 0.950*env_var("_BLOCK_SCALE")  )
    env_sto( "_BAYES_RAD_L2_M" , 0.900*env_var("_BLOCK_SCALE")  )
    # env_sto( "_BAYES_RAD_L2_M" , 0.800*env_var("_BLOCK_SCALE")  ) # 2025-03-11: ?? WINNING PARAMS ??
    # env_sto( "_BAYES_RAD_L2_M" , 0.750*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.700*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.650*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.500*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.400*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.350*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.300*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.250*env_var("_BLOCK_SCALE")  ) # 2025-02-24: ?? WINNING PARAMS ??

    env_sto( "_PLACE_XY_ACCEPT", 0.400*env_var("_BLOCK_SCALE")  )
    # env_sto( "_PLACE_XY_ACCEPT", 0.600*env_var("_BLOCK_SCALE")  )

    # env_sto( "_WIDE_XY_ACCEPT" , 0.750*env_var("_BLOCK_SCALE")  )
    env_sto( "_WIDE_XY_ACCEPT" , 0.900*env_var("_BLOCK_SCALE")  )


    env_sto( "_WIDE_COLLIDE"   , env_var("_BAYES_RAD_L2_M")  ) # 2025-02-25: ?? WINNING PARAMS ??

    # env_sto( "_WIDE_COLLIDE"   , 0.450*env_var("_BLOCK_SCALE")  )
    # env_sto( "_WIDE_COLLIDE"   , 0.625*env_var("_BLOCK_SCALE")  )
    # env_sto( "_WIDE_COLLIDE"   , 0.800*env_var("_BLOCK_SCALE")  )
    # env_sto( "_WIDE_COLLIDE"   , 0.950*env_var("_BLOCK_SCALE")  )
    # env_sto( "_WIDE_COLLIDE"   , 1.125*env_var("_BLOCK_SCALE")  ) # WHY WOULD I EVEN DO THAT?

    env_sto( "_WIDE_PLACEMENT" , 3.000*env_var("_BLOCK_SCALE") ) # 2025-02-25: ?? WINNING PARAMS ??


    env_sto( "_WIDE_Z_ABOVE", 1.75*env_var("_BLOCK_SCALE") )

    env_sto( "_ROBOT_FREE_SPEED", 0.125 * 3.5 ) 
    env_sto( "_ROBOT_HOLD_SPEED", 0.125 * 2.0 )
    env_sto( "_ROBOT_LIN_ACCEL" , 0.500 * 1.5 )

    env_sto( "_ACCEPT_POSN_ERR" , 0.60*env_var( "_BLOCK_SCALE" ) ) # 0.75 # 0.90
    
    env_sto( "_GOAL_GRB" ,
        ( 'and',
            ('GraspObj', 'grnBlock' , _trgtGrn  ), # ; Tower
            ('Supported', 'redBlock', 'grnBlock'), 
            ('Supported', 'bluBlock', 'redBlock'), 

            ('HandEmpty',),
        )
    )

    env_sto( "_GOAL_RRR" ,
        ( 'and',
            ('GraspObj', 'redBlock' , _trgtGrn  ), # ; Tower
            ('Supported', 'redBlock', 'redBlock'), 
            ('Supported', 'redBlock', 'redBlock'), 
            ('HandEmpty',),
        )
    )

    env_sto( "_UPDATE_PERIOD_S", 3.0       ) 
    env_sto( "_OBJ_TIMEOUT_S"  , 60.0*10.0 )

    env_sto( "_SCORE_FILTER_EXP", 0.85 )

    # env_sto( "_UPDATE_FRAC", 0.25 )
    env_sto( "_UPDATE_FRAC", 0.35 )
    # env_sto( "_UPDATE_FRAC", 0.45 )
    # env_sto( "_UPDATE_FRAC", 0.85 )

    # env_sto( "_NULL_EVIDENCE" , True )
    env_sto( "_NULL_EVIDENCE" , False ) # 2025-02-25: ?? WINNING PARAMS ??


    env_sto( "_REPAIR_BAYES" , True ) 


    env_sto( "_DEF_NULL_SCORE", 1.00 )

    # env_sto( "_NULL_THRESH"   , 0.50 )
    env_sto( "_NULL_THRESH"   , 0.60 )
    # env_sto( "_NULL_THRESH"   , 0.65 )
    # env_sto( "_NULL_THRESH"   , 0.75 ) # 2025-02-24: ?? WINNING PARAMS ??
    # env_sto( "_NULL_THRESH"   , 0.95 )

    env_sto( "_GRASP_NUDGE_M", -0.005 )

    env_sto( "_USE_POSE_CHEAT", True )
    


def basic_BT_run( btAction ):
    """ Run a basic BT with `BT_Runner` defaults """
    btr = BT_Runner( btAction, env_var("_BT_UPDATE_HZ"), env_var("_BT_ACT_TIMEOUT_S") )
    btr.setup_BT_for_running()

    while not btr.p_ended():
        btr.tick_once()
        btr.per_sleep()        



########## BT Execution ############################################################################

class BTRunnerwPeriodicScan:
    """ Wrapper for the BT runner that stops for periodic updates """


    def __init__( self, rootBH : Plan, scanInterval_s, scan_cb, check_cb, shot_cb, sleepTime_s = 1.0 ):
        """ Set up perodic scans """
        self.root    = rootBH
        self.period  = scanInterval_s
        self.scanCB  = scan_cb
        self.checkCB = check_cb
        self.shotCB  = shot_cb
        self.runner  = BT_Runner( rootBH, env_var("_BT_UPDATE_HZ"), env_var("_BT_ACT_TIMEOUT_S") )
        self.tSleep  = sleepTime_s
        


    def p_pause_OK( self ):
        """ Is it okay to pause the BT? """
        return ("pause" in str( self.root.current_child.__class__.__name__ ).lower())
        

    def updating_BT_run( self ):
        """ Stop the BT to run callbacks to check the distribution """
        self.lstStop = now()
        self.runner.setup_BT_for_running()

        while not self.runner.p_ended():
            elapsed = now() - self.lstStop
            if elapsed >= self.period:
                if self.p_pause_OK():
                    self.runner.pause()
                    sleep( self.tSleep )
                    if self.shotCB():
                        self.scanCB()
                        if not self.checkCB():
                            self.runner.set_fail( "Distribution does NOT support this action!" )
                    self.runner.resume()
                    sleep( self.tSleep )
                self.lstStop = now()
            self.runner.tick_once()
            self.runner.per_sleep()   



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

        self.robot : UR5_Interface = UR5_Interface( provide_gripper = True ) if (not noBot) else None

        self.memory  = Memory( self.robot, self.perc ) 
        self.cheater = PoseCheater()

        self.symPln = SymPlanner(
            os.path.join( os.path.dirname( __file__ ), "pddl", "domain.pddl" ),
            os.path.join( os.path.dirname( __file__ ), "pddl", "stream.pddl" ),
            planParser = ReactivePlanParser( self.robot )
        )
        self.blcMod = BlockFunctions( self.symPln )
        if (not noBot):
            self.robot.start()
            self.perc.start_vision()

        self.nPlnFl = 0
        self.lmFail = 5


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


    def dummy_object( self ):
        """ Use as target for first-pass sensory planning """
        return GraspObj(
            labels = {'grnBlock':1.0,}, 
            pose   = BASE_TARGET(), 
            ts     = now(), 
            count  = 1, 
            score  = 0.0,
        )


    ##### Task Planning Phases ############################################


    ##### Phase 0 ################################

    def phase_0_Setup( self, symbols ):
        """ Push init symbols """
        self.cheater.log_symbols( symbols )


    ##### Phase 1 ################################

    def phase_1_Perceive( self, Append = False, suppressDeterm = False ):
        """ Take in evidence and form beliefs """

        camPose  = self.robot.get_cam_pose()

        obsrv, metadata = self.perc.segment( _QUERIES )

        self.memory.history.append( msg = "Annotation", datum = {"Event": "The robot takes a 3D picture of the scene."} )
        self.memory.history.append( msg = "ObsMeta"   , datum = metadata )
        
        self.memory.process_observations( 
            obsrv,
            camPose,
            Append
        ) 

        if not suppressDeterm:
            self.memory.get_current_most_likely()


    ##### Phase 2 ################################

    def narrate_state( self ):
        """ Describe the state in English sentences """
        rtnDesc = list()
        for fact in self.blcMod.planner.facts:
            if fact[0] == 'GraspObj':
                rtnDesc.append( f"There is a {_BLOCK_DESC[ fact[1] ]} near to the robot." )
            elif fact[0] == 'Supported':
                rtnDesc.append( f"The {_BLOCK_DESC[ fact[1] ]} is on the {_BLOCK_DESC[ fact[2] ]}." )
            elif fact[0] == 'Blocked':
                rtnDesc.append( f"The {_BLOCK_DESC[ fact[1] ]} cannot be moved until the block above it is moved." )
            else:
                print( f"There is no annotation for predicate: {fact}" )
        return rtnDesc
            

    def phase_2_Conditions( self ):
        """ Get the necessary initial state, Check for goals already met """
        self.symPln.symbols = self.memory.get_current_most_likely()
        if env_var("_USE_POSE_CHEAT"):
            self.cheater.repair_symbol_poses( self.symPln.symbols )

        # self.memory.locate_all( self.symPln.symbols )

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

        self.memory.history.append( msg = "Annotation", datum = {
            "Event": "The robot has taken a 3D picture of the scene.",
            "Desc": self.narrate_state()
        } )
        

    ##### Phase 3 ################################

    def narrate_plan( self ):
        """ Describe the plan in English sentences """
        rtnDesc  = list()
        if self.symPln.status is not Status.FAILURE:
            pdlsPlan = self.blcMod.planner.currPlan[:]
            for action in pdlsPlan:
                actName  = action.name
                actArgs  = action.args
                if actName == "move_free":
                    rtnDesc.append( f"The robot arm will move." )
                elif actName in ("pick", "unstack",):
                    # ?label ?pose ?prevSupport
                    label, pose, prevSupport = actArgs
                    rtnDesc.append( f"The robot arm will pick up the {_BLOCK_DESC[ label ]} from the {_BLOCK_DESC[ prevSupport ]}." )
                elif actName == "move_holding":
                    # ?poseBgn ?poseEnd ?label
                    poseBgn, poseEnd, label = actArgs
                    rtnDesc.append( f"The robot arm will move the {_BLOCK_DESC[ label ]}." )
                elif actName in ("place", "stack",):
                    # ?label ?pose ?support
                    label, pose, support = actArgs
                    rtnDesc.append( f"The robot arm will place the {_BLOCK_DESC[ label ]} on the {_BLOCK_DESC[ support ]}." )
                else:
                    print( f"There is no annotation for action: {action}" )
        else:
            rtnDesc.append( f"The robot arm will attempt to separate blocks that caused the planner to FAIL." )
        return rtnDesc
    

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

            # self.status = Status.FAILURE
            self.status = Status.RUNNING
            self.blcMod.HACK_space_repair_plan( self.robot )

            self.memory.history.append( msg = "Planning Failure" )
            print( f"Planning Failure!" )
            self.memory.history.append( msg = "Annotation", datum = {
                "Event": "The robot has failed to plan any actions.",
            } )
        elif (self.symPln.status == Status.SUCCESS):
            self.status = Status.RUNNING
            print( f"\n\nPlanner thinks we SUCCEEDED!\n\n" )
            self.memory.history.append( msg = "Annotation", datum = {
                "Event": "The robot has determined that the sybolic goal has been met.",
            } )

        if self.blcMod.planner.nxtAct is not None:
            self.memory.history.append( msg = "Annotation", datum = {
                "Event": "The robot has planned a series of actions.",
                "Desc": self.narrate_plan()
            } )
        else:
            self.memory.history.append( msg = "Annotation", datum = {
                "Event": "The robot has failed to plan any actions.",
            } )


    def fetch_src_label_and_pose( self ):
        """ Get the label and initial pose for the current action """
        objName = None
        objPose = None
        for action in self.symPln.nxtAct.children:
            name_i = str( action.__class__.__name__ ).lower()
            if ("pick" in name_i) or ("unstack" in name_i):
                objName = action.args[0]
                objPose = action.args[1]
                break
        return objName, objPose
    

    def fetch_dst_label_and_pose( self ):
        """ Get the label and initial pose for the current action """
        objName = None
        objPose = None
        for action in self.symPln.nxtAct.children:
            name_i = str( action.__class__.__name__ ).lower()
            if ("place" in name_i) or ("stack" in name_i):
                objName = action.args[0]
                objPose = action.args[1]
                break
        return objName, objPose
    

    def check_current_KL_OK( self ):
        """ Find out where we expect important symbols and run the check """

        if env_var("_USE_GRAPHICS"):
            self.memory.plot_KL_history_for_all_obj()

        objName, objPose = self.fetch_src_label_and_pose()
        if objName is None:
            return False
        # return self.memory.check_KL_for_symbol_at_pose( objPose, objName )
        return self.memory.klTr.check_KL_criteria( objName )
    

    def p_OK_to_take_shot( self ):
        """ Return true if we are not too close to the table """
        tcpPose = self.robot.get_tcp_pose()
        camPose = np.dot( tcpPose, self.robot.camXform )
        zzMag   = camPose[2,2]
        rtnShot = False

        # 1. If downward-facing, then Check for correction
        if (zzMag < 0.0):
            hndZdir = vec_unit( np.dot( camPose, np.array( [0.0, 0.0, -1.0, 1.0,] ) )[0:3] )
            camPosn = camPose[0:3,3]
            XYintrc = line_intersect_plane( camPosn, hndZdir, [0.0, 0.0, 0.0,], [0.0, 0.0, 1.0,], pntParallel = False )
            if XYintrc is None:
                rtnShot = False
            else:
                dShot = np.linalg.norm( np.subtract( XYintrc, camPosn ) )
                if dShot >= env_var("_MIN_CAM_PCD_DIST_M"):
                    rtnShot = True
                else:
                    rtnShot = False
        else:
            rtnShot = False
        return rtnShot
    

    def lock_successful_placement( self, btPlan : Plan, symbols : list[GraspObj] ):
        """ Don't allow the Bayes update to nudge the block """
        # FIXME: THINK OF A WAY TO DO THIS THAT DOES NOT BREAK YOUR MODEL
        pass



    def phase_4_Execute_Action( self ):
        """ Attempt to execute the first action in the symbolic plan """

        self.memory.history.append( msg = f"BT BEGIN: {now()}" )

        if _RESPONSIVE_MODE:

            btr = BTRunnerwPeriodicScan( 
                self.symPln.nxtAct, 
                env_var("_UPDATE_PERIOD_S"), 
                self.phase_1_Perceive, 
                self.check_current_KL_OK, 
                self.p_OK_to_take_shot, 
                sleepTime_s = 0.75 
            )

            btr.updating_BT_run()

            _, srcPose = self.fetch_src_label_and_pose()
            _, dstPose = self.fetch_dst_label_and_pose()
            if (btr.runner.status == Status.FAILURE):
                self.status = Status.FAILURE
                if 0:
                    self.memory.history.append( msg = f"Action Failure: {btr.runner.msg}" )
                    self.memory.fail_symbol( srcPose )
                else:
                    self.memory.reset_memory()
            else:
                self.status = Status.RUNNING
                self.memory.move_symbol_from_to_pose( srcPose, dstPose )
        
        else:
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
                    self.memory.history.append( msg = f"Action Failure: {btr.msg}, {now()}" )
                else:
                    # self.memory.history.append( msg = f"Running ... {currTip}, {str(btr.status)}" )
                    self.status = Status.RUNNING

                btr.per_sleep()

            self.memory.history.append( msg = f"BT END: {btr.status}" )

            if (btr.status == Status.FAILURE):
                self.memory.history.append( msg = "Annotation", datum = {
                    "Event": "The robot's plan was not executed correctly.",
                } )
            elif (btr.status == Status.SUCCESS):
                if env_var("_USE_POSE_CHEAT"):
                    self.cheater.log_successful_action( self.blcMod.planner.nxtAct )
                self.memory.history.append( msg = f"Action Success: {btr.msg}, {now()}" )
                self.memory.history.append( msg = "Annotation", datum = {
                    "Event": "The robot's plan was executed correctly.",
                } )
            else:
                self.memory.history.append( msg = "Annotation", datum = {
                    "Event": "The final outcome of the robot's actions was undetermined.",
                } )


    def phase_5_Return_Home( self, goPose = None ):
        """ Get ready for next iteration while updating beliefs """
        if goPose is None:
            goPose = _SAFE
        self.robot.moveL( goPose, 
                          linSpeed = env_var("_ROBOT_FREE_SPEED"),
                          linAccel = env_var("_ROBOT_LIN_ACCEL" ),
                          asynch = False )
        

    ##### Task Planner Main Loop ##########################################

    def names_of_planned_symbols( self ):
        """ Get the names of the symbols that matter """
        rtnLst = list()
        for bhv in self.symPln.nxtAct.children:
            for arg in bhv.args:
                if isinstance( arg, GraspObj ):
                    rtnLst.append( {
                        'label': arg.label,
                        'pose':  arg.pose,
                    } )
        return rtnLst


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

            print( f"\n\n### Iteration {i+1} ###" )
            
            i += 1

            ##### Phase 1 ########################

            print( f"Phase 1, {self.status} ..." )

            # for bgnPose in beginPlanPose:

            # bgnPoses = self.memory.plan_3d_shots( beginPlanPose[0] )
            bgnPoses = self.memory.plan_3d_shots( extract_pose_as_homog( self.dummy_object() ) )

            if not _RESPONSIVE_MODE:
                self.memory.reset_memory()

            # if env_var("_USE_GRAPHICS"):
            #     if _RESPONSIVE_MODE:
            #         if len( self.memory.bMem.beliefs ):
            #             symLst = self.memory.get_current_most_likely()
            #         else:
            #             symLst = self.symPln.symbols
            #         render_memory_list( syms = symLst, robotPose = bgnPoses )
            #     else:
            #         vispy_geo_list_window( [table_geo(),], robotPose = bgnPoses )

            for bgnPose in bgnPoses:
                self.robot.moveL( _SAFE, 
                                  linSpeed = env_var("_ROBOT_FREE_SPEED"),
                                  linAccel = env_var("_ROBOT_LIN_ACCEL" ),
                                  asynch = False )
                self.robot.moveL( bgnPose, 
                                  linSpeed = env_var("_ROBOT_FREE_SPEED"),
                                  linAccel = env_var("_ROBOT_LIN_ACCEL" ),
                                  asynch = False ) # 2024-07-22: MUST WAIT FOR ROBOT TO MOVE            
                self.phase_1_Perceive( Append = True, suppressDeterm = True )

            if env_var("_USE_GRAPHICS"):
                render_scan_list( self.memory.scan )


            ##### Phase 2 ########################

            print( f"Phase 2, {self.status} ..." )
            self.phase_2_Conditions()

            if env_var("_VERBOSE"):
                print(f"Checking goals ...")

            if self.symPln.validate_goal_noisy( self.symPln.goal ):
                self.memory.history.append( msg = f"Believe Success, Iteration {i}: Noisy facts indicate goal was met!\n{self.symPln.facts}" )
                print( f"!!! Noisy success at iteration {i} !!!" )
                self.status = Status.SUCCESS

            if self.status in (Status.SUCCESS, Status.FAILURE):
                print( f"LOOP, {self.status} ..." )
                continue

            if env_var("_USE_GRAPHICS"):
                render_memory_list( syms = self.symPln.symbols )

            ##### Phase 3 ########################

            print( f"Phase 3, {self.status} ..." )
            self.phase_3_Plan_Task()

            if self.p_failed():
                self.memory.reset_memory()
                self.nPlnFl += 1
                print( f"PDLS planner has FAILED {self.nPlnFl} times!" )
                if self.nPlnFl >= self.lmFail:
                    print( f"HALT!: PDLS failure limit ({self.lmFail}) has been REACHED!\n>>> !END! <<<\n" )
                    break
            else:
                self.nPlnFl = 0

            if self.status in (Status.SUCCESS, Status.FAILURE):
                print( f"LOOP, {self.status} ..." )
                continue


            ##### Phase 4 ########################

            print( f"Phase 4, {self.status} ..." )

            self.phase_4_Execute_Action()

            if self.p_failed():
                self.robot.open_gripper()
                

            ##### Phase 5 ########################

            print( f"Phase 5, {self.status} ..." )
            self.phase_5_Return_Home( _SAFE )

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

    
        