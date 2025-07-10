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
from random import random
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
from aspire.env_config import env_var
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

from State import PoseCheater
from Memory import Memory
# from LUMP import LUMP
from draw_beliefs import render_memory_list, render_scan_list
from env_config import set_experiment_env
from utils import deep_copy_memory_list


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
        self.perc  : Perception_OWLv2 = Perception_OWLv2()
        self.robot : UR5_Interface = UR5_Interface( provide_gripper = True ) if (not noBot) else None
        if (not noBot):
            self.robot.start()
            self.perc.start_vision()

        self.memory  = Memory( self.robot ) 
        self.lump    = self.memory.mp
        self.logger  = self.memory.history
        self.cheater = PoseCheater(
            fix_labels = env_var("_CHEAT_LABEL"),
            fix_poses  = env_var("_CHEAT_POSE" )
        )

        self.symPln = SymPlanner(
            os.path.join( os.path.dirname( __file__ ), "pddl", "domain.pddl" ),
            os.path.join( os.path.dirname( __file__ ), "pddl", "stream.pddl" ),
            planParser = ReactivePlanParser( self.robot )
        )
        self.blcMod = BlockFunctions( self.symPln )
        
        self.robot.set_move_callback( self.move_report_cb )

        self.lump.init_object_search( 
            senseCB = self.perception_cb, 
            rMoveCB = self.cam_move_cb  , 
            fetchCB = self.beliefs_cb   , 
            checkCB = self.symbols_present_cb,
            noVizCB = self.scale_vision_thresh_cb
        )

        self.nPlnFl = 0
        self.lmFail = 5


    ##### Callbacks #######################################################

    def move_report_cb( self ):
        """ Record where the robot is at the end of each move """
        self.memory.history.append( 
            msg   = "RobotState", 
            datum = {
                'pose': self.robot.get_tcp_pose().tolist(),
                'q'   : self.robot.get_joint_angles().tolist(),
            }
        )
        self.lump.set_state_from_robot() # WARNING: WILL THIS ACTUALLY IMPROVE LUMP PERFORMANCE?

    
    def perception_cb( self ):
        """ Trigger perception """
        self.phase_1_Perceive( Append = True, suppressDeterm = True )


    def cam_move_cb( self, movPose ):
        """ Position camera for perception """
        self.robot.moveL( _SAFE, 
                          linSpeed = env_var("_ROBOT_FREE_SPEED"),
                          linAccel = env_var("_ROBOT_LIN_ACCEL" ),
                          asynch   = False )
        self.robot.moveL( movPose, 
                          linSpeed = env_var("_ROBOT_FREE_SPEED"),
                          linAccel = env_var("_ROBOT_LIN_ACCEL" ),
                          asynch   = False )


    def beliefs_cb( self ):
        """ Get current beliefs """
        return deep_copy_memory_list( self.memory.bMem.beliefs[:] )
    

    def symbols_present_cb( self ):
        """ Did we find all the symbols? """
        self.phase_2_Conditions()
        return self.symPln.check_goal_objects()
    

    def scale_vision_thresh_cb( self, factor ):
        """ Adjust thresh """
        return self.perc.scale_thresh_by_factor( factor )


    ##### Utils ###########################################################

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


    ##### Phase 1 ################################

    def phase_1_Perceive( self, Append = False, suppressDeterm = False ):
        """ Take in evidence and form beliefs """

        camPose  = self.robot.get_cam_pose()

        obsrv, metadata = self.perc.segment( _QUERIES )

        self.memory.history.append( msg = "Annotation", datum = {"Event": "The robot takes a 3D picture of the scene."} )
        self.memory.history.append( msg = "ObsMeta"   , datum = metadata )
        self.memory.history.append( msg = "Observation BEGIN" )
        
        self.memory.process_observations( 
            obsrv,
            camPose,
            Append
        ) 

        if not suppressDeterm:
            self.memory.get_current_most_likely()

        self.memory.history.append( msg = "Observation END" )
        


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
                pass
                # print( f"There is no annotation for predicate: {fact}" )
        return rtnDesc
            

    def phase_2_Conditions( self ):
        """ Get the necessary initial state, Check for goals already met """
        
        # self.symPln.symbols = self.memory.get_current_most_likely()
        self.symPln.symbols = self.memory.get_current_most_likely( self.symPln.get_goal_objects() )

        if env_var("_USE_POSE_CHEAT"):
            print( f"\n>>>! POSE CHEAT !<<<\n" )
            print( f"\nBefore cheat..." )
            for obj in self.symPln.symbols:
                print( f"\t{obj}" )

            _always_cheat = True
            if (not self.cheater.trouble) or _always_cheat:
                self.symPln.symbols = self.cheater.repair_symbol_poses( self.symPln.symbols )
            else:
                print( f"Cheater in TROUBLE! No nudge!", end = '    ' )

        else:
            print( f"No cheating allowed!", end = '    ' )

        if env_var("_VERBOSE"):
            if env_var("_USE_POSE_CHEAT"):
                print( f"\nAfter cheat..." )
            else:
                print( f"\nStarting Objects:" )
            for obj in self.symPln.symbols:
                print( f"\t{obj}" )

        # self.memory.locate_all( self.symPln.symbols )

        if len( self.symPln.symbols ):
            self.status = Status.RUNNING
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

            self.cheater.log_failed_action( np.eye(4), np.eye(4) )
            
            if env_var("_USE_PERC_HACK"):
                self.lump.log_failed_perc()

            if env_var("_USE_SPACE_HACK"):
                self.blcMod.HACK_space_repair_plan( self.robot )

            self.memory.history.append( msg = "Planning Failure" )
            print( f"Planning Failure!" )
            self.memory.history.append( msg = "Annotation", datum = {
                "Event": "The robot has failed to plan any actions.",
            } )
        elif (self.symPln.status == Status.SUCCESS):
            
            if env_var("_USE_PERC_HACK"):
                self.lump.log_success_perc()
            self.status = Status.SUCCESS
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
                # break
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
                # break
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
            if self.symPln.nxtAct is None:
                if env_var("_USE_PERC_HACK"):
                    self.lump.log_failed_perc()
                print( f"\nNO plan to run!\n" )
                return None
            else:
                if env_var("_USE_PERC_HACK"):
                    self.lump.log_success_perc()
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

            _, srcPose = self.fetch_src_label_and_pose()
            _, dstPose = self.fetch_dst_label_and_pose()

            if (btr.status == Status.FAILURE):
                self.memory.history.append( msg = "Annotation", datum = {
                    "Event": "The robot's plan was not executed correctly.",
                } )

                if env_var("_USE_POSE_CHEAT"):
                    self.cheater.log_failed_action( srcPose, dstPose )
                self.memory.bMem.action_failure_update( srcPose, dstPose )
            elif (btr.status == Status.SUCCESS):
                
                if env_var("_USE_POSE_CHEAT"):
                    self.cheater.log_successful_action( srcPose, dstPose )
                self.memory.bMem.action_success_update( srcPose, dstPose )

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
        self.move_report_cb()
        
        self.symPln.set_goal( env_var("_GOAL_GRB") )
        # self.symPln.set_goal( env_var("_GOAL_OR_RGB") )

        self.cheater.log_symbols( [env_var(f"_KNOWN_BLOCK_{i}") for i in range(3)] )

        self.memory.history.append( msg = "Task Start" )

        while (self.status != Status.SUCCESS) and (i < maxIter): # and (not self.PANIC):
            
            self.status = Status.RUNNING

            print( f"\n\n### Iteration {i+1} ###" )
            
            i += 1

            ##### Phase 1 ########################

            print( f"Phase 1, {self.status} ..." )

            

            if self.cheater.trouble:
                symLst = self.cheater.last_known_symbols()
                symLst.extend( self.cheater.last_known_beliefs() )
                if not _RESPONSIVE_MODE:
                    self.memory.reset_memory()
                self.lump.run_object_search( symLst )
            else:
                if not _RESPONSIVE_MODE:
                    self.memory.reset_memory()
                bgnPoses = self.lump.plan_object_shots( self.cheater.last_known_symbols(), 1.25*self.lump.dShot, 3 )
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
                # print( bgnPoses )
                render_memory_list( 
                    objs = self.memory.bMem.beliefs,
                    syms = self.cheater.last_known_symbols(), 
                    robotPose = bgnPoses 
                )

            if env_var("_SHOW_SEGMENT"):
                self.logger.visualize_last_segmentation()
            

            self.cheater.log_beliefs( self.memory.bMem.beliefs )

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
                self.cheater.trouble = True
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

_TROUBLESHOOT = 0


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
            xHi = -0.469
            yLo =  0.258
            pad =  0.500
            planner.memory.mp.register_aabb_obstacle( [
                [xHi    , yLo    , 0.000,],
                [xHi-pad, yLo+pad, 0.300,],
            ] )
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

    
        