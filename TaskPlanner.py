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
from magpie.BT import Open_Gripper
from magpie import ur5 as ur5
from magpie.poses import repair_pose
# import open3d as o3d

### ASPIRE ###
from aspire.env_config import env_var
from aspire.symbols import ( ObjPose, GraspObj, extract_pose_as_homog, euclidean_distance_between_symbols )
from aspire.utils import ( DataLogger, diff_norm,  )
from aspire.actions import ( display_PDLS_plan, get_BT_plan_until_block_change, BT_Runner, 
                             Interleaved_MoveFree_and_PerceiveScene, MoveFree, GroundedAction, )
from aspire.BlocksTask import set_blocks_env, rand_table_pose

### ASPIRE::PDDLStream ### 
from aspire.pddlstream.pddlstream.utils import read, INF, get_file_path
from aspire.pddlstream.pddlstream.language.generator import from_gen_fn, from_test
from aspire.pddlstream.pddlstream.language.constants import print_solution, PDDLProblem
from aspire.pddlstream.pddlstream.algorithms.meta import solve
from aspire.SymPlanner import SymPlanner

### Local ###
from EROM import EROM



########## PERCEPTION ##############################################################################

class Perception:
    """ Wrapper for the perception model """

    def __init__( self ):
        """ Load the model """
        pass



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
        self.memory = EROM() # Entropy-Ranked Object Memory


    def reset_state( self ):
        """ Erase problem state """
        self.status = Status.INVALID # Running status


    def __init__( self, noBot = False ):
        """ Create a pre-determined collection of poses and plan skeletons """
        set_blocks_env()
        self.datLin = list() # Data to write
        self.outFil = None
        self.noBot  = noBot
        self.outDir = "data/"
        self.reset_memory()
        self.reset_state()
        self.open_file()
        self.perc   = Perception()
        self.robot  = ur5.UR5_Interface() if (not noBot) else None
        self.logger = DataLogger() if (not noBot) else None
        self.symPln = SymPlanner()
        if (not noBot):
            self.robot.start()


    def shutdown( self ):
        """ Stop the Perception Process and the UR5 connection """
        self.dump_to_file( openNext = False )
        if not self.noBot:
            self.robot.stop()


    def p_failed( self ):
        """ Has the system encountered a failure? """
        return (self.status == Status.FAILURE)
    


    ##### Task Planning Phases ############################################

    def phase_1_Perceive( self, Nscans = 1, xform = None ):
        """ Take in evidence and form beliefs """

        for _ in range( Nscans ):
            self.perceive_scene( xform ) # We need at least an initial set of beliefs in order to plan

        self.beliefs = self.merge_and_reconcile_object_memories( cutScoreFrac = env_var("_CUT_MERGE_S_FRAC") )
        self.symbols = self.most_likely_objects( self.beliefs, 
                                                 method = "clean-dupes-score",  # clean-dupes # clean-dupes-score # unique
                                                 cutScoreFrac = env_var("_CUT_SCORE_FRAC") )
        # HACK: TOP OFF THE SCORES OF THE LKG ENTRIES THAT BECAME SYMBOLS
        self.reify_chosen_beliefs( self.world.memory, self.symbols )
        
        
        # if _RECORD_SYM_SEQ:
        #     self.datLin.append( self.capture_object_memory() )

        self.status  = Status.RUNNING

        if env_var("_VERBOSE"):
            print( f"\nStarting Objects:" )
            for obj in self.symbols:
                print( f"\t{obj}" )
            if not len( self.symbols ):
                print( f"\tNO OBJECTS DETERMINIZED" )


    def allocate_table_swap_space( self, Nspots = env_var("_N_XTRA_SPOTS") ):
        """ Find some open poses on the table for performing necessary swaps """
        rtnFacts  = []
        freeSpots = []
        occuSpots = [ extract_pose_as_homog( sym ) for sym in self.symbols]
        while len( freeSpots ) < Nspots:
            nuPose = rand_table_pose()
            print( f"\t\tSample: {nuPose}" )
            collide = False
            for spot in occuSpots:
                if euclidean_distance_between_symbols( spot, nuPose ) < ( env_var("_MIN_SEP") ):
                    collide = True
                    break
            if not collide:
                freeSpots.append( ObjPose( nuPose ) )
                occuSpots.append( nuPose )
        for objPose in freeSpots:
            rtnFacts.extend([
                ('Waypoint', objPose,),
                ('Free', objPose,),
                ('PoseAbove', objPose, 'table'),
            ])
        return rtnFacts
                

    def phase_2_Conditions( self ):
        """ Get the necessary initial state, Check for goals already met """
        
        if not self.check_goal_objects( self.goal, self.symbols ):
            self.logger.log_event( "Required objects missing", str( self.symbols ) )   
            self.status = Status.FAILURE
        else:
            
            self.facts = [ ('Base', 'table',) ] 

            ## Copy `Waypoint`s present in goals ##
            for g in self.goal[1:]:
                if g[0] == 'GraspObj':
                    self.facts.append( ('Waypoint', g[2],) )
                    if abs( extract_pose_as_homog(g[2])[2,3] - env_var("_BLOCK_SCALE")) < env_var("_ACCEPT_POSN_ERR"):
                        self.facts.append( ('PoseAbove', g[2], 'table') )

            ## Ground the Blocks ##
            for sym in self.symbols:
                self.facts.append( ('Graspable', sym.label,) )

                blockPose = self.get_grounded_fact_pose_or_new( sym )

                # print( f"`blockPose`: {blockPose}" )
                self.facts.append( ('GraspObj', sym.label, blockPose,) )
                if not self.p_grounded_fact_pose( blockPose ):
                    self.facts.append( ('Waypoint', blockPose,) )

            ## Fetch Relevant Facts ##
            self.facts.extend( self.ground_relevant_predicates_noisy() )

            ## Populate Spots for Block Movements ##, 2024-04-25: Injecting this for now, Try a stream later ...
            self.facts.extend( self.allocate_table_swap_space( env_var("_N_XTRA_SPOTS") ) )

            if env_var("_VERBOSE"):
                print( f"\n### Initial Symbols ###" )
                for sym in self.facts:
                    print( f"\t{sym}" )
                print()


    def phase_3_Plan_Task( self ):
        """ Attempt to solve the symbolic problem """

        self.task = self.pddlstream_from_problem()

        self.logger.log_event( "Begin Solver" )

        # print( dir( self.task ) )
        if 0:
            print( f"\nself.task.init\n" )
            pprint( self.task.init )
            print( f"\nself.task.goal\n" )
            pprint( self.task.goal )
            print( f"\nself.task.domain_pddl\n" )
            pprint( self.task.domain_pddl )
            print( f"\nself.task.stream_pddl\n" )
            pprint( self.task.stream_pddl )

        try:
            
            solution = solve( 
                self.task, 
                algorithm      = "adaptive", #"focused", #"binding", #"incremental", #"adaptive", 
                unit_costs     = True, # False, #True, 
                unit_efforts   = True, # False, #True,
                reorder        = True,
                initial_complexity = 2,
                # max_complexity = 4,
                # max_failures  = 4,
                # search_sample_ratio = 1/4

            )

            print( "Solver has completed!\n\n\n" )
            print_solution( solution )
            
        except Exception as ex:
            self.logger.log_event( "SOLVER FAULT", format_exc() )
            self.status = Status.FAILURE
            print_exc()
            solution = (None, None, None)
            self.noSoln += 1 # DEATH MONITOR

        plan, cost, evaluations = solution

        if (plan is not None) and len( plan ):
            display_PDLS_plan( plan )
            self.currPlan = plan
            self.action   = get_BT_plan_until_block_change( plan, self, env_var("_UPDATE_PERIOD_S") )
            self.noSoln   = 0 # DEATH MONITOR
        else:
            self.noSoln += 1 # DEATH MONITOR
            self.logger.log_event( "NO SOLUTION" )
            self.status = Status.FAILURE


    def phase_4_Execute_Action( self ):
        """ Attempt to execute the first action in the symbolic plan """
        
        btr = BT_Runner( self.action, env_var("_BT_UPDATE_HZ"), env_var("_BT_ACT_TIMEOUT_S") )
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

            btr.per_sleep()

        self.logger.log_event( "BT END", str( btr.status ) )

        print( f"Did the BT move a reading?: {self.world.move_reading_from_BT_plan( self.action )}" )


    def phase_5_Return_Home( self, goPose ):
        """ Get ready for next iteration while updating beliefs """
        btAction = GroundedAction( args = list(), robot = self.robot, name = "Return Home" )
        btAction.add_children([
            Open_Gripper( ctrl = self.robot ),
            Interleaved_MoveFree_and_PerceiveScene( 
                MoveFree( [None, ObjPose( goPose )], robot = self.robot, suppressGrasp = True ), 
                self, 
                env_var("_UPDATE_PERIOD_S"), 
                initSenseStep = True 
            ),
        ])
        
        btr = BT_Runner( btAction, env_var("_BT_UPDATE_HZ"), env_var("_BT_ACT_TIMEOUT_S") )
        btr.setup_BT_for_running()

        while not btr.p_ended():
            btr.tick_once()
            btr.per_sleep()

        print( f"\nRobot returned to \n{goPose}\n" )
        


    def p_fact_match_noisy( self, pred ):
        """ Search grounded facts for a predicate that matches `pred` """
        for fact in self.facts:
            if pred[0] == fact[0]:
                same = True 
                for i in range( 1, len( pred ) ):
                    if type( pred[i] ) != type( fact[i] ):
                        same = False 
                        break
                    elif isinstance( pred[i], str ) and (pred[i] != fact[i]):
                        same = False
                        break
                    elif (pred[i].index != fact[i].index):
                        same = False
                        break
                if same:
                    return True
        return False

    
    def validate_goal_noisy( self, goal ):
        """ Check if the system believes the goal is met """
        if goal[0] == 'and':
            for g in goal[1:]:
                if not self.p_fact_match_noisy( g ):
                    return False
            return True
        else:
            raise ValueError( f"Unexpected goal format!: {goal}" )


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

        self.reset_beliefs() 
        self.reset_state() 
        self.set_goal()
        self.logger.begin_trial()

        indicateSuccess = False
        t5              = now()

        self.robot.moveL( beginPlanPose, asynch = False ) # 2024-07-22: MUST WAIT FOR ROBOT TO MOVE

        while (self.status != Status.SUCCESS) and (i < maxIter): # and (not self.PANIC):

            
            # sleep(1)

            print( f"### Iteration {i+1} ###" )
            
            i += 1

            ##### Phase 1 ########################

            print( f"Phase 1, {self.status} ..." )

            self.set_goal()

            camPose = self.robot.get_cam_pose()

            expBgn = now()
            if (expBgn - t5) < env_var("_UPDATE_PERIOD_S"):
                sleep( env_var("_UPDATE_PERIOD_S") - (expBgn - t5) )
            
            self.phase_1_Perceive( 1, camPose )
            
            # if _USE_GRAPHICS:
            #     self.display_belief_geo()

            ##### Phase 2 ########################

            print( f"Phase 2, {self.status} ..." )
            self.phase_2_Conditions()

            if self.validate_goal_noisy( self.goal ):
                indicateSuccess = True
                self.logger.log_event( "Believe Success", f"Iteration {i}: Noisy facts indicate goal was met!\n{self.facts}" )
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

            # DEATH MONITOR
            if self.noSoln >= self.nonLim:
                self.logger.log_event( "SOLVER BRAINDEATH", f"Iteration {i}: Solver has failed {self.noSoln} times in a row!" )
                break

            if self.p_failed():
                print( f"LOOP, {self.status} ..." )
                continue

            ##### Phase 4 ########################

            print( f"Phase 4, {self.status} ..." )
            t4 = now()
            if (t4 - expBgn) < env_var("_UPDATE_PERIOD_S"):
                sleep( env_var("_UPDATE_PERIOD_S") - (t4 - expBgn) )
            self.phase_4_Execute_Action()

            ##### Phase 5 ########################

            print( f"Phase 5, {self.status} ..." )
            t5 = now()
            if (t5 - t4) < env_var("_UPDATE_PERIOD_S"):
                sleep( env_var("_UPDATE_PERIOD_S") - (t5 - t4) )
            self.phase_5_Return_Home( beginPlanPose )

            print()

        if 0: #self.PANIC:
            print( "\n\nWARNING: User-requested shutdown or other fault!\n\n" )
            self.logger.end_trial(
                False,
                {'PANIC': True, 'end_symbols' : list( self.symbols ), }
            )
        else:
            self.logger.end_trial(
                indicateSuccess,
                {'end_symbols' : list( self.symbols ) }
            )

        

        self.logger.save( "data/Baseline" )

        print( f"\n##### PLANNER END with status {self.status} after iteration {i} #####\n\n\n" )






########## EXPERIMENT HELPER FUNCTIONS #############################################################

_GOOD_VIEW_POSE = None
_HIGH_VIEW_POSE = None

def responsive_experiment_prep( beginPlanPose = None ):
    """ Init system and return a ref to the planner """
    # planner = BaselineTaskPlanner()
    planner = TaskPlanner()
    print( planner.robot.get_tcp_pose() )

    if beginPlanPose is None:
        if env_var("_BLOCK_SCALE") < 0.030:
            beginPlanPose = _GOOD_VIEW_POSE
        else:
            beginPlanPose = _HIGH_VIEW_POSE

    planner.robot.open_gripper()
    # sleep( OWL_init_pause_s )
    return planner




    

########## MAIN ####################################################################################

_TROUBLESHOOT   = 0
_VISION_TEST    = 0
_EXP_BGN_POSE   = _HIGH_VIEW_POSE
_HIGH_TWO_POSE  = None


_CONF_CAM_POSE_ANGLED1 = repair_pose( np.array( [[ 0.55 , -0.479,  0.684, -0.45 ],
                                                 [-0.297, -0.878, -0.376, -0.138],
                                                 [ 0.781,  0.003, -0.625,  0.206],
                                                 [ 0.   ,  0.   ,  0.   ,  1.   ],] ) )

_YCB_LANDSCAPE_CLOSE_BGN = repair_pose( np.array( [[-0.698,  0.378,  0.608, -0.52 ],
                                                   [ 0.264,  0.926, -0.272, -0.308],
                                                   [-0.666, -0.029, -0.746,  0.262],
                                                   [ 0.   ,  0.   ,  0.   ,  1.   ],] ) )



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
            rbt = ur5.UR5_Interface()
            rbt.start()
            sleep(2)
            print( f"Began at pose:\n{rbt.get_tcp_pose()}" )
            
            # rbt.moveL( repair_pose( _GOOD_VIEW_POSE ), asynch = False )
            # sleep(5)
            # rbt.moveL( repair_pose( _HIGH_VIEW_POSE ), asynch = False )
            
            rbt.stop()


    elif _VISION_TEST:
        print( f"########## Running Vision Pipeline Test at {dateStr} ##########" )

        planner = responsive_experiment_prep( _HIGH_VIEW_POSE )
        
        print( f"\nAt Pose 1:\n{_HIGH_VIEW_POSE}\n" )
        planner.world.perc.capture_image()
        sleep(1)
        
        planner.robot.moveL( _HIGH_TWO_POSE, asynch = False )
        print( f"\nAt Pose 2:\n{_HIGH_TWO_POSE}\n" )
        planner.world.perc.capture_image()
        sleep(1)

        observs = planner.world.perc.merge_and_build_model()
        xfrmCam = planner.robot.get_cam_pose()
        planner.world.full_scan_noisy( xfrmCam, observations = observs )
        # if _USE_GRAPHICS:
        #     planner.memory.display_belief_geo( planner.world.scan )

        planner.world.rectify_readings( copy_readings_as_LKG( planner.world.scan ) )
        observs = planner.world.get_last_best_readings()
        planner.world.full_scan_noisy( xfrmCam, observations = observs )
        # if _USE_GRAPHICS:
        #     planner.memory.display_belief_geo( planner.world.scan )

        sleep( 2.5 )
        planner.shutdown()


    else:
        print( f"########## Running Planner at {dateStr} ##########" )

        try:
            planner = responsive_experiment_prep( _YCB_LANDSCAPE_CLOSE_BGN ) # _EXP_BGN_POSE
            planner.solve_task( maxIter = 30, beginPlanPose = _YCB_LANDSCAPE_CLOSE_BGN )
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

    
        