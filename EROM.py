########## INIT ####################################################################################

import time
now = time.time
from math import isnan

import numpy as np
from py_trees.common import Status

from aspire.env_config import env_var
from aspire.utils import match_name, normalize_dist
from aspire.symbols import ( ObjPose, GraspObj, extract_pose_as_homog, p_symbol_inside_workspace_bounds,
                             euclidean_distance_between_symbols )
from aspire.actions import GroundedAction

from magpie_control.poses import translation_diff

### Local ###
from Bayes import ObjectMemory
from utils import snap_z_to_nearest_block_unit_above_zero, set_quality_score, mark_readings_LKG



########## HELPER FUNCTIONS ########################################################################


def hacked_offset_map( pose ) -> np.ndarray:
    """ Calculate a hack to the pose """
    hackXfrm = np.eye(4)
    offset   = np.zeros( (3,) )
    vec      = pose[0:3,3]
    # midX     = env_var("_MIN_X_OFFSET") + env_var("_X_WRK_SPAN")*0.75
    midY     = env_var("_MIN_Y_OFFSET") + env_var("_Y_WRK_SPAN")*0.50
    # lesY     = midY - env_var("_Y_WRK_SPAN")*0.25
    minY     = env_var("_MIN_Y_OFFSET")
    maxY     = env_var("_MAX_Y_OFFSET")
    height   = 0.5*env_var("_BLOCK_SCALE")
    hackMap  = [ [[env_var("_MIN_X_OFFSET"), midY, height], [-2.0/100.0, -0.5/100.0, 0.0]],
                 [[env_var("_MAX_X_OFFSET"), minY, height], [-1.0/100.0, -1.5/100.0, 0.0]], 
                 [[env_var("_MAX_X_OFFSET"), maxY, height], [-1.0/100.0, +1.0/100.0, 0.0]],]
    weights = list()
    for hack in hackMap:
        weights.append( 1.0 / np.linalg.norm( np.subtract( vec, hack[0] ) ) )
    tot = sum( weights )
    for i, hack in enumerate( hackMap ):
        offset += (weights[i]/tot) * np.array( hack[1] )
    hackXfrm[0:3,3] = offset
    return hackXfrm


def observation_to_readings( obs : dict, xform = None ):
    """ Parse the Perception Process output struct """
    rtnBel = []
    if xform is None:
        xform = np.eye(4)
    for item in obs.values():
        dstrb = {}
        tScan = item['Time']

        for nam, prb in item['Probability'].items():
            dstrb[ match_name( nam ) ] = prb
        if env_var("_NULL_NAME") not in dstrb:
            dstrb[ env_var("_NULL_NAME") ] = env_var("_CONFUSE_PROB")

        dstrb = normalize_dist( dstrb )

        if len( item['Pose'] ) == 16:
            hackXfrm = hacked_offset_map( xform.dot( np.array( item['Pose'] ).reshape( (4,4,) ) )  )
            xform    = hackXfrm.dot( xform ) #env_var("_HACKED_OFFSET").dot( xform )
            objPose  = xform.dot( np.array( item['Pose'] ).reshape( (4,4,) ) ) 
            # HACK: SNAP TO NEAREST BLOCK UNIT && SNAP ABOVE TABLE
            objPose[2,3] = snap_z_to_nearest_block_unit_above_zero( objPose[2,3] )
        else:
            raise ValueError( f"`observation_to_readings`: BAD POSE FORMAT!\n{item['Pose']}" )
        
        # Attempt to quantify how much we trust this reading
        

        rtnObj = GraspObj( 
            labels = dstrb, 
            pose   = ObjPose( objPose ), 
            ts     = tScan, 
            count  = item['Count'], 
            score  = 0.0 
        )
        set_quality_score( rtnObj )

        rtnBel.append( rtnObj )
    return rtnBel



########## OBJECT PERMANENCE #######################################################################

def cut_bottom_fraction( objs : list[GraspObj], frac ):
    """ Return a version of `objs` with the bottom `frac` scores removed """
    if len( objs ) > 1:
        rtnObjs = sorted( objs, key = lambda item: item.score, reverse = True )
        remNum  = int( frac * len( rtnObjs ) )
        # print( f"Cutting {frac} = {remNum}/{len(rtnObjs)} objects!" )
        if remNum > 0:
            return rtnObjs[ 0:-remNum ]
        else:
            return rtnObjs[:]
    else:
        return objs[:]


def rectify_readings( objReadingList : list[GraspObj], useTimeout = True ):
    """ Accept/Reject/Update noisy readings from the system, used by both LKG and EROM """
    tCurr  = now()
    nuMem  = list()
    nuSet  = set([])
    rmSet  = set([])
    totLst = objReadingList[:]

    Ntot = len( totLst )
    # 1. For every item of [incoming object info + previous info]
    for r, objR in enumerate( totLst ):
        
        # HACK: ONLY CONSIDER OBJECTS INSIDE THE WORKSPACE
        if not p_symbol_inside_workspace_bounds( extract_pose_as_homog( objR ) ):
            print( f"Object {objR} is outside the workspace!" )
            continue

        if (useTimeout and ((tCurr - objR.ts) > env_var("_OBJ_TIMEOUT_S"))):
            continue

        # 3. Search for a collision with existing info
        conflict = [objR,]
        for m in range( r+1, Ntot ):
            objM = totLst[m]
            if euclidean_distance_between_symbols( objR, objM ) < env_var("_LKG_SEP"):
                conflict.append( objM )

        print( f"Conflict: {len(conflict)}" )

        # 4. Sort overlapping indications and add only the top
        conflict.sort( key = lambda item: item.score, reverse = True )
        top    = conflict[0]
        nuHash = id( top )
        if (nuHash not in nuSet) and (nuHash not in rmSet):
            nuMem.append( top )
            nuSet.add( nuHash )
            if len( conflict ) > 1:
                rmSet.update( set( [id(elem) for elem in conflict[1:]] ) )
            print( f"Added {top} to the new memory!" )
    print( f"Rectified {len(nuMem)} objects!" )
    return nuMem


def merge_and_reconcile_object_memories( belLst : list[GraspObj], lkgLst : list[GraspObj], tau = None ):
    """ Calculate a consistent object state from LKG Memory and Beliefs """
    rtnLst = list()

    if tau is None:
        tau = env_var("_SCORE_DECAY_TAU_S")
    mrgLst  = list()
    mrgLst.extend( belLst )
    mrgLst.extend( lkgLst )
    tCurr   = now()
    
    print( f"Number to decay: {len(mrgLst)}" )
    
    # Filter and Decay stale readings
    for r in mrgLst:
        score_r = np.exp( -(tCurr - r.ts) / tau ) * r.score
        if isnan( score_r ):
            print( f"\nWARN: Got a NaN score with count {r.count}, distribution {r.labels}, and age {tCurr - r.ts}\n" )
            score_r = 0.0
        r.score = score_r
        rtnLst.append( r )

    print( f"Number to reconcile: {len(rtnLst)}" )
    
    # Enforce consistency and return
    return rectify_readings( rtnLst, env_var("_USE_TIMEOUT") )


def objs_choose_k( objs, k : int, bgn = None, end = None ):
    """ Length choose k """
    ## Init ##
    Nreadings = len( objs )
    comboList = []
    if bgn is None:
        bgn = 0
    if end is None:
        end = Nreadings-(k-1)
    ## Generate all reading combinations ##
    if k == 1:
        for i in range( bgn, end ):
            comboList.append( [objs[i],] )
    elif k > 1:
        for i in range( bgn, end ):
            lst_i = [objs[i],]
            for add_j in objs_choose_k( objs, k-1, bgn+1, end+1 ):
                lst_j = lst_i[:]
                lst_j.extend( add_j )
                comboList.append( lst_j )
    return comboList


def gen_combos( objs : list[GraspObj], idx = 0 ):
    comboList = []
    if (idx+1) >= len(objs):
        # for label_i, prob_i in objs[ idx ].labels.items():
        # for label_i in env_var("_BLOCK_NAMES"):
        for label_i in sorted( list( objs[ idx ].labels.keys() ) ):
            prob_i = objs[ idx ].labels[ label_i ]
            comboList.append( [
                GraspObj( label = label_i, pose  = objs[ idx ].pose, 
                          prob  = prob_i , score = objs[ idx ].score, labels = None,
                          ts = objs[ idx ].ts , count = objs[ idx ].count ),
            ] )
    else:
        # for label_i, prob_i in objs[ idx ].labels.items():
        # for label_i in env_var("_BLOCK_NAMES"):
        for label_i in sorted( list( objs[ idx ].labels.keys() ) ):
            prob_i = objs[ idx ].labels[ label_i ]
            obj_i = GraspObj( label = label_i, pose  = objs[ idx ].pose, 
                               prob  = prob_i , score = objs[ idx ].score, labels = None,
                               ts    = objs[ idx ].ts , count = objs[ idx ].count )
            for cmb_j in gen_combos( objs, idx+1 ):
                lst_j = [obj_i,]
                lst_j.extend( cmb_j )
                comboList.append( lst_j )
    return comboList



def most_likely_objects( objList : list[GraspObj], k : int, method = "unique-non-null", cutScoreFrac = 0.5 ):
    """ Get the `N` most likely combinations of object classes """
    
    ### Drop Worst Readings ###
    if (1.0 > cutScoreFrac > 0.0):
        objs = cut_bottom_fraction( objList, cutScoreFrac )
    else:
        objs = objList[:]

    ### Combination Generator ###
    def gen_combos_top( objs : list[GraspObj], k : int ):
        totalList = []
        kGroups   = objs_choose_k( objs, k )
        for o_i in kGroups:
            totalList.extend( gen_combos( o_i ) )
        return totalList
    
    def prod( lst : list[GraspObj] ):
        p = 1.0
        for itm in lst:
            p *= itm.prob
        return p

    # totCombos = gen_combos_top( objs , k )
    totCombos = [item for item in gen_combos_top( objs , k ) if prod(item) > 0.001]
    totCombos.sort(
        key     = lambda x: prod(x),
        reverse = 1
    )

    print( f"\nTotal Combos: {len(totCombos)}" )
    for combo in totCombos:
        print( f"Combo: {prod(combo):.3f}, {combo}" )
    print()

    ### Filtering Methods ###

    def p_unique_labels( objs : list[GraspObj] ):
        """ Return true if there are as many classes as there are objects """
        lbls = set([sym.label for sym in objs])
        return len( lbls ) == len( objs )
    
    def p_unique_non_null_labels( objs : list[GraspObj] ):
        """ Return true if there are as many classes as there are objects """
        lbls = set([sym.label for sym in objs])
        if env_var("_NULL_NAME") in lbls: 
            return False
        return len( lbls ) == len( objs )
    
    def clean_dupes_prob( objLst : list[GraspObj] ):
        """ Return a version of `objLst` with duplicate objects removed """
        dctMax = {}
        for sym in objLst:
            if not sym.label in dctMax:
                dctMax[ sym.label ] = sym
            elif sym.prob > dctMax[ sym.label ].prob:
                dctMax[ sym.label ] = sym
        return list( dctMax.values() )
    
    def clean_dupes_score( objLst : list[GraspObj] ):
        """ Return a version of `objLst` with duplicate objects removed """
        dctMax = {}
        for sym in objLst:
            if not sym.label in dctMax:
                dctMax[ sym.label ] = sym
            elif sym.score > dctMax[ sym.label ].score:
                dctMax[ sym.label ] = sym
        return list( dctMax.values() )

    ### Apply the chosen Filtering Method to all possible combinations ###

    rtnSymbols = list()

    if (method == "unique"):
        for combo in totCombos:
            if p_unique_labels( combo ):
                rtnSymbols = combo
                break
    elif (method == "unique-non-null"):
        for combo in totCombos:
            if p_unique_non_null_labels( combo ):
                rtnSymbols = combo
                break
    elif (method == "clean-dupes"):
        rtnSymbols = clean_dupes_prob( totCombos[0] )
    elif (method == "clean-dupes-score"):
        rtnSymbols = clean_dupes_score( totCombos[0] )
    else:
        raise ValueError( f"`most_likely_objects`: Filtering method \"{method}\" is NOT recognized!" )
    
    ### Return all non-null symbols ###
    # rtnLst = [sym for sym in rtnSymbols if sym.label != env_var("_NULL_NAME")]
    print( f"\nDeterminized {len(rtnSymbols)} objects!\n" )
    return rtnSymbols


def reify_chosen_beliefs( objs : list[GraspObj], chosen, factor = env_var("_REIFY_SUPER_BEL") )->None:
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
                # obj.score = maxSc * factor
                obj.ts    = now()



########## ENTROPY-RANKED OBJECT MEMORY ############################################################

class EROM:
    """ Entropy-Ranked Object Memory """

    def reset_memory( self ):
        """ Erase memory components """
        self.scan : list[GraspObj]  = list()
        self.beliefs = ObjectMemory()
        self.LKG    : list[GraspObj]    = list()
        self.ranked : list[GraspObj] = list()
        # self.symbols: list[GraspObj] = list()


    def __init__( self ):
        self.reset_memory()


    def process_observations( self, obs, xform = None ):
        """ Integrate one noisy scan into the current beliefs """

        # self.scan = observation_to_readings( obs, xform )
        self.scan = cut_bottom_fraction( observation_to_readings( obs, xform ), env_var("_CUT_INTAKE_S_FRAC") )
        # self.scan = rectify_readings( 
        #     cut_bottom_fraction( observation_to_readings( obs, xform ), env_var("_CUT_INTAKE_S_FRAC") ), 
        #     useTimeout = False 
        # )
        # self.scan = rectify_readings( 
        #     observation_to_readings( obs, xform ), 
        #     useTimeout = False 
        # )
        

        # LKG and Belief are updated SEPARATELY and merged LATER as symbols
        # self.LKG     = rectify_readings( copy_readings_as_LKG( self.scan ) )
        # self.LKG.extend( copy_readings_as_LKG( self.scan ) )
        self.LKG.extend( self.scan )
        mark_readings_LKG( self.LKG, val = True )
        self.LKG = rectify_readings( self.LKG, useTimeout = env_var("_USE_TIMEOUT") )
        self.beliefs.belief_update( self.scan, xform, maxRadius = env_var("_MAX_UPDATE_RAD_M") )


    def rank_combined_memory( self ):
        """ Reconcile and rank the two memory streams """

        self.LKG = cut_bottom_fraction( self.LKG, env_var("_CUT_LKG_S_FRAC") )

        # self.ranked = sorted( 
        #     merge_and_reconcile_object_memories( 
        #         list( self.beliefs.beliefs ), 
        #         list( self.LKG ), 
        #         tau          = env_var("_SCORE_DECAY_TAU_S"), 
        #     ), 
        #     key = lambda item: item.score, 
        #     reverse = True 
        # )

        self.ranked = merge_and_reconcile_object_memories( 
            list( self.beliefs.beliefs ), 
            list( self.LKG ), 
            tau          = env_var("_SCORE_DECAY_TAU_S"), 
        )

        self.ranked = cut_bottom_fraction( self.ranked, env_var("_CUT_MERGE_S_FRAC") )
        print( f"Cut Ranking: {len(self.ranked)}" )

        return self.ranked
    

    # def migrate_LKG_to_beliefs( self ):
    #     """ Move applicable LKG to the belief memory """
    #     nuLKG = list()
    #     for lkg_i in self.LKG:
    #         if not self.beliefs.integrate_one_reading( lkg_i, camXform = None, suppressNew = True ):
    #             nuLKG.append( lkg_i )
    #     self.LKG = nuLKG
        
        
    def get_current_most_likely( self ):
        """ Generate symbols """

        # self.migrate_LKG_to_beliefs()
        self.rank_combined_memory()

        print( f"Beliefs: {len(self.beliefs.beliefs)}, LKG: {len(self.LKG)}, Total: {len(self.ranked)}" )

        print( f"\nRanked: {len(self.ranked)}" )
        for obj in self.ranked:
            print( obj )
        print()

        rtnLst = most_likely_objects( 
            self.ranked, 
            env_var("_N_REQD_OBJS"),
            method       = "unique-non-null", # "unique", #"unique-non-null", 
            cutScoreFrac = env_var("_CUT_DETERM_S_FRAC")
        )
        # reify_chosen_beliefs( self.ranked, rtnLst, factor = env_var("_REIFY_SUPER_BEL") )

        # print( rtnLst )
        # self.symbols = rtnLst

        return rtnLst
    

    def move_reading_from_BT_plan( self, planBT : GroundedAction ):
        """ Infer reading to be updated by the robot action, Then update it """
        _verbose = True
        # NOTE: This should run after a BT successfully completes
        # NOTE: This function exits after the first object move
        # NOTE: This function assumes that the reading nearest to the beginning of the 
        updated = False
        dMin    = 1e9
        endMin  = None
        objMtch = list()

        # self.ranked = rectify_readings( self.ranked, useTimeout = False )
        
        for act_i in planBT.children:
            if "MoveHolding" in act_i.__class__.__name__:
                poseBgn, poseEnd, label = act_i.args
                
                # BAD WARNING ERROR: RANKED LIST HAS ALREADY BEEN CUT, NEED TO MOVE BOTH LISTS

                for objM in self.ranked:
                    dist_ij = euclidean_distance_between_symbols( objM, poseBgn )
                    if (dist_ij <= env_var("_MIN_SEP")) and (dist_ij < dMin) and (label in objM.labels):
                        dMin    = dist_ij
                        endMin  = poseEnd
                        updated = True
                        objMtch.append( objM )
                # break
        if updated:
            if planBT.status == Status.SUCCESS:
                for obj_m in objMtch:
                    obj_m.pose = endMin
                    obj_m.ts   = now() # 2024-07-27: THIS IS EXTREMELY IMPORTANT ELSE THIS READING DIES --> BAD BELIEFS
                    obj_m.score *= env_var('_SCORE_DIV_FAIL') 
                    # obj_m.score = env_var('_SCORE_BIGNUM') 
                    # 2024-07-27: NEED TO DO SOME DEEP THINKING ABOUT THE FRESHNESS OF RELEVANT FACTS
                    if _verbose:
                        print( f"`get_moved_reading_from_BT_plan`: BT {planBT.name} updated {obj_m}!" )  
            else:
                for obj_m in objMtch:
                    # obj_m.score /= env_var('_SCORE_DIV_FAIL')
                    obj_m.score = 0.0
        else:
            if _verbose:
                print( f"`get_moved_reading_from_BT_plan`: NO update applied by BT {planBT.name}!" )    

        return updated


    


    
    

    
    
    
    