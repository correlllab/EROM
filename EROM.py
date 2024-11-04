########## INIT ####################################################################################

import time
now = time.time
from math import isnan

import numpy as np
from py_trees.common import Status

from aspire.env_config import env_var
from aspire.utils import match_name, normalize_dist, get_confused_class_reading
from aspire.symbols import ( ObjPose, GraspObj, extract_pose_as_homog, p_symbol_inside_workspace_bounds,
                             euclidean_distance_between_symbols )
from aspire.actions import GroundedAction

from magpie_control.poses import translation_diff

### Local ###
from Bayes import ObjectMemory
from utils import ( snap_z_to_nearest_block_unit_above_zero, set_quality_score, mark_readings_LKG, 
                    LogPickler, deep_copy_memory_list )



########## HELPER FUNCTIONS ########################################################################


def hacked_offset_map( pose ) -> np.ndarray:
    """ Calculate a hack to the pose """
    hackXfrm = np.eye(4)
    offset   = np.zeros( (3,) )
    vec      = pose[0:3,3]
    
    minX     = env_var("_MIN_X_OFFSET")
    midX     = env_var("_MIN_X_OFFSET") + env_var("_X_WRK_SPAN")*0.50
    maxX     = env_var("_MAX_X_OFFSET")
    
    minY     = env_var("_MIN_Y_OFFSET")
    midY     = env_var("_MIN_Y_OFFSET") + env_var("_Y_WRK_SPAN")*0.50
    maxY     = env_var("_MAX_Y_OFFSET")

    # lesY     = midY - env_var("_Y_WRK_SPAN")*0.25
    minY     = env_var("_MIN_Y_OFFSET")
    maxY     = env_var("_MAX_Y_OFFSET")
    height   = 0.5*env_var("_BLOCK_SCALE")
    hackMap  = [ [[minX, minY, height], [ 1.0/100.0, 1.0/100.0, 0.0]],
                 [[minX, maxY, height], [ 1.0/100.0, 0.0/100.0, 0.0]],
                 [[midX, midY, height], [ 1.0/100.0, 1.0/100.0, 0.0]],
                 [[maxX, minY, height], [ 1.0/100.0, 1.0/100.0, 0.0]], 
                 [[maxX, maxY, height], [ 1.0/100.0, 0.0/100.0, 0.0]],]
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
            if prb > 0.0001:
                dstrb[ match_name( nam ) ] = prb
            else:
                dstrb[ match_name( nam ) ] = env_var("_CONFUSE_PROB")
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
            score  = 0.0,
            cpcd   = item['CPCD'],
        )
        set_quality_score( rtnObj )

        rtnBel.append( rtnObj )
    return rtnBel



########## OBJECT PERMANENCE #######################################################################

def cut_bottom_fraction( objs : list[GraspObj], frac ):
    """ Return a version of `objs` with the bottom `frac` scores removed, Also return removed list """
    N = len( objs )
    if N:
        rtnObjs = sorted( objs, key = lambda item: item.score, reverse = True )
        remNum  = int( frac * N )
        keepNm  = N - remNum

        if remNum > 0:
            return rtnObjs[ 0:-remNum ], rtnObjs[ keepNm: ]
        else:
            return rtnObjs[:], list()
    else:
        return objs[:], list()


def rectify_readings( objReadingList : list[GraspObj], useTimeout = True ):
    """ Accept/Reject/Update noisy readings from the system, used by both LKG and EROM """
    tCurr  = now()
    nuMem  = list()
    rmMem  = list()
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

    for r, objR in enumerate( totLst ):
        if id( objR ) in rmSet:
            rmMem.append( objR )

    return nuMem, rmMem


def merge_and_reconcile_object_memories( belLst : list[GraspObj], lkgLst : list[GraspObj], tau = None ):
    """ Calculate a consistent object state from LKG Memory and Beliefs """
    rtnLst = list()

    if tau is None:
        tau = env_var("_SCORE_DECAY_TAU_S")
    mrgLst  = list()
    mrgLst.extend( belLst )
    mrgLst.extend( lkgLst )
    tCurr   = now()
    
    if env_var("_USE_DECAY"):
        print( f"\nNumber to decay: {len(mrgLst)}" )
    
    # Filter and Decay stale readings
    for r in mrgLst:
        if env_var("_USE_DECAY"):
            factor = np.exp( -(tCurr - r.ts) / tau )
            print( f"Decay factor: {factor:.3f}" )
            score_r = factor * r.score
            if isnan( score_r ):
                print( f"\nWARN: Got a NaN score with count {r.count}, distribution {r.labels}, and age {tCurr - r.ts}\n" )
                score_r = 0.0
            r.score = score_r
        rtnLst.append( r )

    if env_var("_USE_DECAY"):
        print()

    print( f"Number to reconcile: {len(rtnLst)}" )
    
    # Enforce consistency and return
    return rectify_readings( rtnLst, env_var("_USE_TIMEOUT") )
    # recLst, _ = rectify_readings( rtnLst, env_var("_USE_TIMEOUT") )
    # return recLst


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
                          ts = objs[ idx ].ts , count = objs[ idx ].count,
                          parent = objs[ idx ] ),
            ] )
    else:
        # for label_i, prob_i in objs[ idx ].labels.items():
        # for label_i in env_var("_BLOCK_NAMES"):
        for label_i in sorted( list( objs[ idx ].labels.keys() ) ):
            prob_i = objs[ idx ].labels[ label_i ]
            obj_i = GraspObj( label = label_i, pose  = objs[ idx ].pose, 
                              prob  = prob_i , score = objs[ idx ].score, labels = None,
                              ts    = objs[ idx ].ts , count = objs[ idx ].count,
                              parent = objs[ idx ] )
            for cmb_j in gen_combos( objs, idx+1 ):
                lst_j = [obj_i,]
                lst_j.extend( cmb_j )
                comboList.append( lst_j )
    return comboList


def mark_parents_by_symbol( symList : list[GraspObj] ):
    """ Mark objects that gave rise to symbols """
    for obj in symList:
        if obj.parent is not None:
            obj.parent.SYM = True


def unmark_symbol_parents( objList : list[GraspObj] ):
    """ Remove the symbol flag from all items """
    for obj in objList:
        obj.SYM = False


def most_likely_objects( objList : list[GraspObj], k : int, 
                         method = "unique-non-null", cutScoreFrac = 0.5 ) -> list[GraspObj]:
    """ Get the `N` most likely combinations of object classes """
    
    ### Drop Worst Readings ###
    if (1.0 > cutScoreFrac > 0.0):
        objs, _ = cut_bottom_fraction( objList, cutScoreFrac )
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

    # print( f"\nTotal Combos: {len(totCombos)}" )
    # for combo in totCombos:
    #     print( f"Combo: {prod(combo):.3f}, {combo}" )
    # print()

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
    
    def p_unique_nonnull_nocollide( objs : list[GraspObj] ):
        """ Return true if there are as many classes as there are objects """
        if not p_unique_non_null_labels( objs ):
            return False
        for i, obj_i in enumerate( objs ):
            for j, obj_j in enumerate( objs ):
                if (i != j) and (euclidean_distance_between_symbols( obj_i, obj_j ) < env_var("_LKG_SEP")):
                    return False
        return True
    
    

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
    elif (method == "unique-nonull-nocollide"):
        for combo in totCombos:
            if p_unique_nonnull_nocollide( combo ):
                rtnSymbols = combo
                break
    else:
        raise ValueError( f"`most_likely_objects`: Filtering method \"{method}\" is NOT recognized!" )
    
    ### Return all non-null symbols ###
    # rtnLst = [sym for sym in rtnSymbols if sym.label != env_var("_NULL_NAME")]
    print( f"\nDeterminized {len(rtnSymbols)} objects!\n" )
    return rtnSymbols, totCombos


def reify_chosen_beliefs( chosen : list[GraspObj] )->None:
    """ Super-believe in the beliefs we believed in. 
        That is: Refresh the timestamp and score of readings that ultimately became grounded symbols """
    nUpdt = 0
    for chsn in chosen:
        if chsn.parent is not None:
            chsn.parent.ts    = now()
            chsn.parent.score *= env_var("_SCORE_MULT_DETERM")
            nUpdt += 1
    print( f"\n### Reified {nUpdt} beliefs! ###\n" )
        


########## ENTROPY-RANKED OBJECT MEMORY ############################################################

class EROM:
    """ Entropy-Ranked Object Memory """

    def reset_memory( self ):
        """ Erase memory components """
        self.scan : list[GraspObj]  = list()
        self.beliefs = ObjectMemory()
        self.LKG    : list[GraspObj]    = list()
        self.ranked : list[GraspObj] = list()
        self.lastPose = dict()


    def __init__( self ):
        self.history = LogPickler( prefix = "EROM-Memories", outDir = "data" )
        self.reset_memory()


    def shutdown( self ):
        """ Save the memory """
        self.history.dump_to_file( openNext = False )


    def process_observations( self, obs, xform = None ):
        """ Integrate one noisy scan into the current beliefs """

        readings = observation_to_readings( obs, xform )
        nuScan, _ = cut_bottom_fraction( readings, env_var("_CUT_INTAKE_S_FRAC") )

        if len( self.scan ) > 0:
            self.scan = nuScan
        else:
            self.scan, _ = rectify_readings( 
                nuScan, 
                useTimeout = False 
            )
        
        self.LKG.extend( self.scan )
        mark_readings_LKG( self.LKG, val = True )
        self.LKG, LKGrec = cut_bottom_fraction( self.LKG, env_var("_CUT_LKG_S_FRAC") )
        self.LKG, LKGcut = rectify_readings( self.LKG, useTimeout = env_var("_USE_TIMEOUT") )
        
        self.beliefs.belief_update( self.scan, xform, maxRadius = env_var("_MAX_UPDATE_RAD_M") )

        self.history.append( 
            datum = {
                "observation" : deep_copy_memory_list( readings             ),
                "scan"        : deep_copy_memory_list( self.scan            ),
                "LKG"         : deep_copy_memory_list( self.LKG             ),
                "LKGrec"      : deep_copy_memory_list( LKGrec               ),
                "LKGcut"      : deep_copy_memory_list( LKGcut               ),
                "beliefs"     : deep_copy_memory_list( self.beliefs.beliefs ),
            },
            msg = "memory" 
        )


    def rank_combined_memory( self ):
        """ Reconcile and rank the two memory streams """

        self.ranked, rankRct = merge_and_reconcile_object_memories( 
            list( self.beliefs.beliefs ), 
            list( self.LKG ), 
            tau = env_var("_SCORE_DECAY_TAU_S"), 
        )

        self.ranked, rankCut = cut_bottom_fraction( self.ranked, env_var("_CUT_MERGE_S_FRAC") )

        print( f"Cut Ranking: {len(self.ranked)}" )

        self.history.append( 
            datum = {
                "ranking" : deep_copy_memory_list( self.ranked ),
                "remRec"  : deep_copy_memory_list( rankRct     ),
                "remCut"  : deep_copy_memory_list( rankCut     ),
            },
            msg = "totalRank" 
        )

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

        rtnLst, combos = most_likely_objects( 
            self.ranked, 
            env_var("_N_REQD_OBJS"),
            method       = "unique-nonull-nocollide", # "unique-non-null", # "unique", #"unique-non-null", 
            cutScoreFrac = env_var("_CUT_DETERM_S_FRAC")
        )

        pairs = list()
        for sym in rtnLst:
            pairs.append({
                'symbol' : sym.deep_copy(),
                'parent' : sym.parent.deep_copy() if (sym.parent is not None) else None,
            })
        stoCmb = list()
        for c in combos:
            stoCmb.append( deep_copy_memory_list( c ) )

        self.history.append( 
            datum = {
                'pairs'  : pairs,
                'combos' : stoCmb,
            },
            msg   = "symbols" 
        )

        combined = self.beliefs.beliefs[:] + self.LKG[:]
        unmark_symbol_parents( combined )
        mark_parents_by_symbol( rtnLst )

        return rtnLst
    

    def check_reading_movement( self ):
        """ Check if any readings have moved """
        combined = self.ranked[:] + self.LKG[:]
        print( "\n##### Readings Moved #####" )
        for objM in combined:
            if id( objM ) in self.lastPose:
                movMag = translation_diff( self.lastPose[ id( objM ) ], extract_pose_as_homog( objM ) )
                if movMag > 0.0001:
                    print( f"{movMag:.3f}[m] movement by {objM}" )
            self.lastPose[ id( objM ) ] = extract_pose_as_homog( objM )
        print()
    

    def move_reading_from_BT_plan( self, planBT : GroundedAction ):
        """ Infer reading to be updated by the robot action, Then update it """
        _verbose = True
        # NOTE: This should run after a BT successfully completes
        # NOTE: This function exits after the first object move
        # NOTE: This function assumes that the reading nearest to the beginning of the 
        updated  = False
        dMin     = 1e9
        endMin   = None
        objMtch  = list()
        combined = self.ranked[:] + self.LKG[:]

        for act_i in planBT.children:
            if "MoveHolding" in act_i.__class__.__name__:
                poseBgn, poseEnd, label = act_i.args

                # for objM in self.ranked:
                for objM in combined:
                    dist_ij = euclidean_distance_between_symbols( objM, poseBgn )

                    # if (dist_ij <= env_var("_MIN_SEP")) and (dist_ij < dMin) and (label in objM.labels):
                    # if (dist_ij <= (env_var("_MAX_UPDATE_RAD_M")*1.5) ) and (dist_ij < dMin) and (label in objM.labels):
                    if (dist_ij <= env_var("_MAX_UPDATE_RAD_M") ) and (dist_ij < dMin) and (label in objM.labels):
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
                    obj_m.score *= env_var('_SCORE_MULT_SUCCESS') 
                    # obj_m.score = env_var('_SCORE_BIGNUM') 
                    # 2024-07-27: NEED TO DO SOME DEEP THINKING ABOUT THE FRESHNESS OF RELEVANT FACTS
                    if _verbose:
                        print( f"`get_moved_reading_from_BT_plan`: BT {planBT.name} updated {obj_m}!" )  

                self.check_reading_movement()

                self.history.append( 
                    datum = deep_copy_memory_list( objMtch ),
                    msg   = "moved" 
                )
                
            else:
                for obj_m in objMtch:
                    # obj_m.score /= env_var('_SCORE_DIV_FAIL')
                    obj_m.score = 0.0
                    nul_m = GraspObj( 
                        label = env_var("_NULL_NAME"), 
                        labels = get_confused_class_reading( env_var("_NULL_NAME"), env_var("_CONFUSE_PROB"), 
                                                             env_var("_BLOCK_NAMES") ),
                        pose = obj_m.pose, 
                        prob = 1.0, 
                        score = env_var("_DEF_NULL_SCORE"), 
                        ts = now(), 
                        count = 1 
                    )
                    for _ in range( env_var("_N_MISS_PUNISH") ):
                        self.beliefs.integrate_one_reading( nul_m, camXform = None, suppressNew = True )

                self.history.append( 
                    datum = deep_copy_memory_list( objMtch ),
                    msg   = "demoted" 
                )
        else:
            if _verbose:
                print( f"`get_moved_reading_from_BT_plan`: NO update applied by BT {planBT.name}!" )    

        return updated, extract_pose_as_homog( endMin )


    


    
    

    
    
    
    