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
from magpie.poses import repair_pose, translation_diff
# import open3d as o3d

### Local ###
from aspire.env_config import env_var
from aspire.symbols import ( ObjPose, GraspObj, extract_pose_as_homog, euclidean_distance_between_symbols )
from aspire.utils import ( DataLogger, diff_norm,  )
from aspire.actions import ( display_PDLS_plan, get_BT_plan_until_block_change, BT_Runner, 
                             Interleaved_MoveFree_and_PerceiveScene, MoveFree, GroundedAction, )
from aspire.BlocksTask import set_blocks_env, rand_table_pose

# ### PDDLStream ### 
from aspire.pddlstream.pddlstream.utils import read, INF, get_file_path
from aspire.pddlstream.pddlstream.language.generator import from_gen_fn, from_test
from aspire.pddlstream.pddlstream.language.constants import print_solution, PDDLProblem
from aspire.pddlstream.pddlstream.algorithms.meta import solve

### Local ###
from Bayes import ObjectMemory



##### Planner #############################################################

class TaskPlanner:
    """ Basic task planning loop """

    ##### File Ops ########################################################

    def open_file( self ):
        """ Set the name of the current file """
        dateStr     = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.outNam = f"Task-Planner_{dateStr}.json"
        self.outFil = open( os.path.join( self.outDir, self.outNam ), 'w' )


    def dump_to_file( self, openNext = False ):
        """ Write all data lines to a file """
        # self.outFil.writelines( [f"{str(line)}\n" for line in self.datLin] )        
        json.dump( self.datLin, self.outFil )
        self.outFil.close()
        if openNext:
            self.datLin = list()
            self.open_file()


    ##### Init ############################################################

    def reset_symbols( self ):
        """ Erase belief memory """
        self.symbols = list() # ------- Determinized beliefs
        self.facts   = list() # ------- Grounded predicates


    def reset_memory( self ):
        """ Erase memory components """
        self.scan    = list()
        self.beliefs = ObjectMemory()
        self.LKG     = list()
        self.ranked  = list()


    def reset_state( self ):
        """ Erase problem state """
        self.status = Status.INVALID # Running status
        self.task   = None # --------- Current task definition
        self.goal   = tuple() # ------ Current goal specification
        self.grasp  = list() # ------- ? NOT USED ?
        self.datLin = list() # ------- Data to write
        self.outDir = "data/"
        self.open_file()


    def __init__( self, world = None, noViz = False, noBot = False ):
        """ Create a pre-determined collection of poses and plan skeletons """
        set_blocks_env()
        self.reset_symbols()
        self.reset_state()
        self.robot  = ur5.UR5_Interface() if (not noBot) else None
        self.logger = DataLogger() if (not noBot) else None
        self.noViz  = noViz
        self.noBot  = noBot
        # DEATH MONITOR
        self.noSoln =  0
        self.nonLim = 10
        if (not noBot):
            self.robot.start()


    def shutdown( self ):
        """ Stop the Perception Process and the UR5 connection """
        self.world.stop()
        self.dump_to_file( openNext = False )
        if not self.noBot:
            self.robot.stop()


    ##### Stream Helpers ##################################################

    def get_grounded_pose_or_new( self, homog ):
        """ If there is a `Waypoint` approx. to `homog`, then return it, Else create new `ObjPose` """
        for fact in self.facts:
            if fact[0] == 'Waypoint' and ( euclidean_distance_between_symbols( homog, fact[1] ) <= env_var("_ACCEPT_POSN_ERR")):
                return fact[1]
        return ObjPose( homog )


    def p_grounded_fact_pose( self, poseOrObj ):
        """ Does this exist as a `Waypoint`? """
        homog = extract_pose_as_homog( poseOrObj )
        for fact in self.facts:
            if fact[0] == 'Waypoint' and ( euclidean_distance_between_symbols( homog, fact[1] ) <= env_var("_ACCEPT_POSN_ERR")):
                return True
        return False



    ##### Task Planning Helpers ###########################################

    def pddlstream_from_problem( self, pdls_stream_map = None ):
        """ Set up a PDDLStream problem with the UR5 """

        domain_pddl  = read( get_file_path( __file__, os.path.join( 'task_planning/', 'domain.pddl' ) ) )
        stream_pddl  = read( get_file_path( __file__, os.path.join( 'task_planning/', 'stream.pddl' ) ) )
        constant_map = {}
        stream_map = pdls_stream_map if ( pdls_stream_map is not None ) else dict()

        if env_var("_VERBOSE"):
            print( "About to create problem ... " )

        return PDDLProblem( domain_pddl, constant_map, stream_pddl, stream_map, self.facts, self.goal )
    

    def set_goal( self, nuGoal ):
        """ Set the goal """

        self.goal = nuGoal
        
        # ( 'and',
            
        #     ('GraspObj', 'grnBlock' , _trgtGrn  ), # ; Tower
        #     ('Supported', 'ylwBlock', 'grnBlock'), 
        #     ('Supported', 'bluBlock', 'ylwBlock'),

        #     ('HandEmpty',),
        # )

        if env_var("_VERBOSE"):
            print( f"\n### Goal ###" )
            pprint( self.goal )
            print()


    def p_failed( self ):
        """ Has the system encountered a failure? """
        return (self.status == Status.FAILURE)
    

    ##### Object Permanence ###############################################

    def merge_and_reconcile_object_memories( self, tau = None, cutScoreFrac = 0.5  ):
        """ Calculate a consistent object state from LKG Memory and Beliefs """
        if tau is None:
            tau = env_var("_SCORE_DECAY_TAU_S")
        mrgLst  = list()
        tCurr   = now()
        totLst  = self.memory.beliefs[:]
        LKGmem  = self.world.get_last_best_readings()
        totLst.extend( LKGmem )

        def cut_bottom_fraction( objs, frac ):
            """ Return a version of `objs` with the bottom `frac` scores removed """
            rtnObjs = sorted( objs, key = lambda item: item.score, reverse = True )
            keepNum  = len( rtnObjs ) - int( frac * len( rtnObjs ) )
            return rtnObjs[ 0:keepNum ]
        
        # Filter and Decay stale readings
        for r in totLst:
            if ((tCurr - r.ts) <= env_var("_OBJ_TIMEOUT_S")):
                score_r = np.exp( -(tCurr - r.ts) / tau ) * r.score
                if isnan( score_r ):
                    print( f"\nWARN: Got a NaN score with count {r.count}, distribution {r.labels}, and age {tCurr - r.ts}\n" )
                    score_r = 0.0
                r.score = score_r
                mrgLst.append( r )

        if (1.0 > cutScoreFrac > 0.0):
            mrgLst = cut_bottom_fraction( mrgLst, cutScoreFrac )
        
        # Enforce consistency and return
        return self.world.rectify_readings( mrgLst, suppressStorage = True )
        

    def most_likely_objects( self, objList, method = "unique-non-null", cutScoreFrac = 0.5 ):
        """ Get the `N` most likely combinations of object classes """
        # FIXME: THIS IS PROBABLY WRONG WAS APPLICABLE TO THE SIMULATION ONLY BUT NOT 
        #        A VARIABLE COLLECTION OF OBJECTS, POSSIBLE SOURCE OF BAD BAD ERRORS

        def cut_bottom_fraction( objs, frac ):
            """ Return a version of `objs` with the bottom `frac` scores removed """
            rtnObjs = sorted( objs, key = lambda item: item.score, reverse = True )
            keepNum  = len( rtnObjs ) - int( frac * len( rtnObjs ) )
            return rtnObjs[ 0:keepNum ]

        ### Combination Generator ###

        def gen_combos( objs ):
            ## Init ##
            if (1.0 > cutScoreFrac > 0.0):
                objs = cut_bottom_fraction( objs, cutScoreFrac )
            comboList = [ [1.0,[],], ]
            ## Generate all class combinations with joint probabilities ##
            for bel in objs:
                nuCombos = []
                for combo_i in comboList:
                    for label_j, prob_j in bel.labels.items():
                        prob_ij = combo_i[0] * prob_j

                        objc_ij = GraspObj( label = label_j, pose = bel.pose, 
                                            prob = prob_j, score = bel.score, labels = bel.labels )
                        
                        nuCombos.append( [prob_ij, combo_i[1]+[objc_ij,],] )
                comboList = nuCombos
            ## Sort all class combinations with decreasing probabilities ##
            comboList.sort( key = (lambda x: x[0]), reverse = True )
            return comboList

        ### Filtering Methods ###

        def p_unique_labels( objs ):
            """ Return true if there are as many classes as there are objects """
            lbls = set([sym.label for sym in objs])
            return len( lbls ) == len( objs )
        
        def p_unique_non_null_labels( objs ):
            """ Return true if there are as many classes as there are objects """
            lbls = set([sym.label for sym in objs])
            if env_var("_NULL_NAME") in lbls: 
                return False
            return len( lbls ) == len( objs )
        
        def clean_dupes_prob( objLst ):
            """ Return a version of `objLst` with duplicate objects removed """
            dctMax = {}
            for sym in objLst:
                if not sym.label in dctMax:
                    dctMax[ sym.label ] = sym
                elif sym.prob > dctMax[ sym.label ].prob:
                    dctMax[ sym.label ] = sym
            return list( dctMax.values() )
        
        def clean_dupes_score( objLst ):
            """ Return a version of `objLst` with duplicate objects removed """
            dctMax = {}
            for sym in objLst:
                if not sym.label in dctMax:
                    dctMax[ sym.label ] = sym
                elif sym.score > dctMax[ sym.label ].score:
                    dctMax[ sym.label ] = sym
            return list( dctMax.values() )

        ### Apply the chosen Filtering Method to all possible combinations ###

        totCombos  = gen_combos( objList )
        rtnSymbols = list()

        if (method == "unique"):
            for combo in totCombos:
                if p_unique_labels( combo[1] ):
                    rtnSymbols = combo[1]
                    break
        elif (method == "unique-non-null"):
            for combo in totCombos:
                if p_unique_non_null_labels( combo[1] ):
                    rtnSymbols = combo[1]
                    break
        elif (method == "clean-dupes"):
            rtnSymbols = clean_dupes_prob( totCombos[0][1] )
        elif (method == "clean-dupes-score"):
            rtnSymbols = clean_dupes_score( totCombos[0][1] )
        else:
            raise ValueError( f"`ResponsiveTaskPlanner.most_likely_objects`: Filtering method \"{method}\" is NOT recognized!" )
        
        ### Return all non-null symbols ###
        rtnLst = [sym for sym in rtnSymbols if sym.label != env_var("_NULL_NAME")]
        print( f"\nDeterminized {len(rtnLst)} objects!\n" )
        return rtnLst
    
    
    def reify_chosen_beliefs( self, objs, chosen, factor = env_var("_REIFY_SUPER_BEL") ):
        """ Super-believe in the beliefs we believed in. 
            That is: Refresh the timestamp and score of readings that ultimately became grounded symbols """
        posen = [ extract_pose_as_homog( ch ) for ch in chosen ]
        maxSc = 0.0
        for obj in objs:
            if obj.score > maxSc:
                maxSc = obj.score
        for obj in objs:
            for cPose in posen:
                if (translation_diff( cPose, extract_pose_as_homog( obj ) ) <= env_var("_LKG_SEP")):
                    obj.score = maxSc * factor
                    obj.ts    = now()
                

    ##### Noisy Task Monitoring ###########################################

    def get_sampled_block( self, label ):
        """ If a block with `label` was sampled, then return a reference to it, Otherwise return `None` """
        for sym in self.symbols:
            if sym.label == label:
                return sym
        return None
    

    def get_grounded_fact_pose_or_new( self, homog ):
        """ If there is a `Waypoint` approx. to `homog`, then return it, Else create new `ObjPose` """ 
        for fact in self.facts:
            if fact[0] == 'Waypoint' and (euclidean_distance_between_symbols( homog, fact[1] ) <= env_var("_ACCEPT_POSN_ERR")):
                return fact[1]
            if fact[0] == 'GraspObj' and (euclidean_distance_between_symbols( homog, fact[2] ) <= env_var("_ACCEPT_POSN_ERR")):
                return fact[2]
        return ObjPose( homog )
    
    
    def ground_relevant_predicates_noisy( self ):
        """ Scan the environment for evidence that the task is progressing, using current beliefs """
        rtnFacts = []
        ## Gripper Predicates ##
        if len( self.grasp ):
            for graspedLabel in self.grasp:
                # [hndl,pDif,bOrn,] = grasped
                # labl = self.world.get_handle_name( hndl )
                rtnFacts.append( ('Holding', graspedLabel,) )
        else:
            rtnFacts.append( ('HandEmpty',) )
        ## Obj@Loc Predicates ##
        # A. Check goals
        for g in self.goal[1:]:
            if g[0] == 'GraspObj':
                pLbl = g[1]
                pPos = g[2]
                tObj = self.get_sampled_block( pLbl )
                if (tObj is not None) and (euclidean_distance_between_symbols( pPos, tObj ) <= env_var("_ACCEPT_POSN_ERR")):
                    rtnFacts.append( g ) # Position goal met
        # B. No need to ground the rest

        ## Support Predicates && Blocked Status ##
        # Check if `sym_i` is supported by `sym_j`, blocking `sym_j`, NOTE: Table supports not checked
        supDices = set([])
        for i, sym_i in enumerate( self.symbols ):
            for j, sym_j in enumerate( self.symbols ):
                if i != j:
                    lblUp = sym_i.label
                    lblDn = sym_j.label
                    posUp = extract_pose_as_homog( sym_i )
                    posDn = extract_pose_as_homog( sym_j )
                    xySep = diff_norm( posUp[0:2,3], posDn[0:2,3] )
                    zSep  = posUp[2,3] - posDn[2,3] # Signed value
                    if ((xySep <= 1.65*env_var("_BLOCK_SCALE")) and (1.65*env_var("_BLOCK_SCALE") >= zSep >= 0.9*env_var("_BLOCK_SCALE"))):
                        supDices.add(i)
                        rtnFacts.extend([
                            ('Supported', lblUp, lblDn,),
                            ('Blocked', lblDn,),
                            ('PoseAbove', self.get_grounded_fact_pose_or_new( posUp ), lblDn,),
                        ])
        for i, sym_i in enumerate( self.symbols ):
            if i not in supDices:
                rtnFacts.extend( [
                    ('Supported', sym_i.label, 'table',),
                    ('PoseAbove', self.get_grounded_fact_pose_or_new( sym_i ), 'table',),
                ] )
        ## Where the robot at? ##
        robotPose = ObjPose( self.robot.get_tcp_pose() )
        rtnFacts.extend([ 
            ('AtPose', robotPose,),
            ('WayPoint', robotPose,),
        ])
        ## Return relevant predicates ##
        return rtnFacts


    def check_goal_objects( self, goal, symbols ):
        """ Return True if the labels mentioned in the goals are a subset of the determinized symbols """
        goalSet = set([])
        symbSet = set( [sym.label for sym in symbols] )
        for g in goal:
            if isinstance( g, (tuple, list) ):
                prdName = g[0]
                if prdName == 'GraspObj':
                    goalSet.add( g[1] )
                elif prdName == 'Supported':
                    goalSet.add( g[1] )
                    goalSet.add( g[2] )
                else:
                    continue
        return (goalSet <= symbSet)
    

    def block_exists( self, label ):
        """ See if a fact already covers this block """
        for f in self.facts:
            if (f[0] == 'GraspObj') and (f[1] == label):
                return True
        return False


    # def capture_object_memory( self ):
    #     """ Save the current state of the entire object permanence framework as nested dictionaries """
    #     return {
    #         'time'    : now(),
    #         'scan'    : copy_readings_dict( self.world.scan ),
    #         'symbols' : copy_readings_dict( self.symbols ),
    #         'LKGmem'  : copy_readings_dict( self.world.memory ),
    #         'beliefs' : copy_readings_dict( self.memory.beliefs ),
    #     }


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

    
        