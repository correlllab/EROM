########## INIT ####################################################################################

##### Imports #####

### Standard ###
import time
now = time.time
from collections import deque, Counter
# from typing import Dict, Deque
from math import log
from uuid import uuid4
from copy import deepcopy

### Special ###
import numpy as np
import matplotlib.pyplot as plt

### Local ###
from magpie_control.poses import vec_unit, translation_diff
from magpie_control.ur5 import UR5_Interface

from aspire.env_config import env_var
from aspire.utils import match_name, normalize_dist
from aspire.symbols import ( ObjPose, GraspObj, extract_pose_as_homog, euclidean_distance_between_symbols )
from aspire.actions.pdls_behaviors import GroundedAction, Plan

from utils import ( LogPickler, zip_dict_sorted_by_decreasing_value, deep_copy_memory_list, )
from OWLv2_Segment import Perception_OWLv2
from Bayes import BayesMemory


##### Constants #####
# _REVERSE_QUERIES = {
#     "bluBlock": {'query': "a photo of a blue block"  , 'abbrv': "blu", },
#     "ylwBlock": {'query': "a photo of a yellow block", 'abbrv': "ylw", },
#     "grnBlock": {'query': "a photo of a green block" , 'abbrv': "grn", },
#     "redBlock": {'query': "a photo of a red block" , 'abbrv': "red", },
# }

_REVERSE_QUERIES = {
    "bluBlock": {'query': "a photo of a small block", 'abbrv': "blu", },
    "ylwBlock": {'query': "a photo of a small block", 'abbrv': "ylw", },
    "grnBlock": {'query': "a photo of a small block", 'abbrv': "grn", },
    "redBlock": {'query': "a photo of a small block", 'abbrv': "red", },
}



########## HELPER FUNCTIONS ########################################################################

def observation_to_readings( obs, xform = None, zOffset = 0.0 ):
    """ Parse the Perception Process output struct """
    rtnBel = []
    if xform is None:
        xform = np.eye(4)

    if isinstance( obs, dict ):
        obs = list( obs.values() )

    for item in obs:
        dstrb = {}
        tScan = item['Time']

        # WARNING: CLASSES WITH A ZERO PRIOR WILL NOT ACCUMULATE EVIDENCE!

        if isinstance( item['Probability'], dict ):
            for nam, prb in item['Probability'].items():
                if prb > 0.0001:
                    dstrb[ match_name( nam ) ] = prb
                else:
                    dstrb[ match_name( nam ) ] = env_var("_CONFUSE_PROB")

            for nam in env_var("_BLOCK_NAMES"):
                if nam not in dstrb:
                    dstrb[ nam ] = env_var("_CONFUSE_PROB")
                
            dstrb = normalize_dist( dstrb )

        if len( item['Pose'] ) == 16:
            objPose = xform.dot( np.array( item['Pose'] ).reshape( (4,4,) ) ) 

            # # HACK: SNAP THE Z-COMPONENT DURING SCAN
            # objPose[2,3] = snap_z_to_nearest_block_unit_above_zero( objPose[2,3] )

            # HACK: PUSH THE BLOCK POSE INTO THE HAND
            objPose[2,3] += env_var("_GRASP_NUDGE_M")

        else:
            raise ValueError( f"`observation_to_readings`: BAD POSE FORMAT!\n{item['Pose']}" )
        
        # Create reading
        rtnObj = GraspObj( 
            labels = dstrb, 
            pose   = ObjPose( objPose ), 
            ts     = tScan, 
            count  = item['Count'], 
            score  = 0.0,
            cpcd   = item['CPCD'],
        )

        # Transform CPCD
        mov = xform.copy()
        mov[2,3] += zOffset
        rtnObj.cpcd.transform( mov )

        # Store mask centroid ray
        rtnObj.meta['rayOrg'] = xform[0:3,3].reshape(3)
        rtnObj.meta['rayDir'] = np.dot( xform[0:3,0:3], item['camRay'].reshape( (3,1,) ) ).reshape(3)

        rtnBel.append( rtnObj )
    return rtnBel


def most_likely_objects( objList : list[GraspObj], method : str | list = "sufficient" ):
    """ Get the `N` most likely combinations of object classes """
    ### Combination Generator ###

    def gen_combos( objs : list[GraspObj] ):
        ## Init ##

        # comboList = [ [1.0,[],], ]
        # comboList = deque([ [1.0,[],], ])
        comboList = deque()

        ## Generate all class combinations with joint probabilities ##
        blkNam = env_var("_ACTUAL_NAMES") # Prevent repeated fetch
        Nnames = len( blkNam )
        Nobjct = len( objList )
        Ncombo = Nnames ** Nobjct

        if env_var("_VERBOSE"):
            print( f"There are {Ncombo} combinations to inspect!" )

        for i in range( Ncombo ):
            prob_i = 1.0
            num_j  = i
            idx_j  = 0
            symLst = [None for _ in range(Nobjct)]
            for j, objct_j in enumerate( objList ):
                num_j, idx_j = divmod( num_j, Nnames )
                label_j = blkNam[ idx_j ]
                prob_j  = objct_j.labels[ label_j ]
                prob_i += -np.log( prob_j ) # Negative log likelihood
                symLst[j] = GraspObj( label = label_j, pose  = objct_j.pose, 
                                      prob  = prob_j , score = objct_j.score, labels = objct_j.labels,
                                      parent = objct_j )
            comboList.appendleft( [prob_i, symLst] )

        ## Sort all class combinations with decreasing probabilities ##
        comboList = list( comboList )
        # comboList.sort( key = (lambda x: x[0]), reverse = True )
        comboList.sort( key = (lambda x: x[0]), reverse = False ) # Negative log likelihood
        return comboList

    ### Filtering Methods ###

    def p_match_label_quantity( objs : list[GraspObj], reqs : list[tuple] ):
        """ Return True if there are at least as many objects of the required type as required """
        reqd = Counter( [item[1] for item in reqs if (item[0] == 'GraspObj')] )
        oDct = Counter( [item.label for item in objs] )
        for k, v in reqd.items():
            if k in oDct:
                if oDct[k] < v:
                    return False
            else:
                return False
        return True
        

    def p_enough_labels( objs : list[GraspObj] ):
        """ Return true if there are as many classes as there are objects """
        lbls = set([sym.label for sym in objs])
        return len( lbls ) >= (len( objs[0].labels )-1)

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
        dctMax: dict[str,GraspObj] = {}
        for sym in objLst:
            if not sym.label in dctMax:
                dctMax[ sym.label ] = sym
            elif sym.prob > dctMax[ sym.label ].prob:
                dctMax[ sym.label ] = sym
        return list( dctMax.values() )
    
    def clean_dupes_score( objLst : list[GraspObj] ):
        """ Return a version of `objLst` with duplicate objects removed """
        dctMax : dict[str,GraspObj] = {}
        for sym in objLst:
            if not sym.label in dctMax:
                dctMax[ sym.label ] = sym
            elif sym.score > dctMax[ sym.label ].score:
                dctMax[ sym.label ] = sym
        return list( dctMax.values() )

    ### Apply the chosen Filtering Method to all possible combinations ###

    totCombos  = gen_combos( objList )
    rtnSymbols = list()

    if isinstance( method, list ):
        found = False
        for combo in totCombos:
            if p_match_label_quantity( combo[1], method ):
                rtnSymbols = combo[1]
                found      = True
                break
        if not found:
            for combo in totCombos:
                lblSet = set([])
                for cSym in combo[1]:
                    lblSet.add( cSym.label )
                if len( lblSet ) > 1:
                    rtnSymbols = combo[1]
                    break
            # rtnSymbols = totCombos[0][1]  
    elif (method == "sufficient"):
        for combo in totCombos:
            if p_enough_labels( combo[1] ):
                rtnSymbols = combo[1]
                break
    elif (method == "unique"):
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
        

def image_offset( image : np.ndarray, bbox : np.ndarray, zLen :float ):
    """ Project a ray through the center of the mask """
    rows   = image.shape[0]
    rwHf   = rows / 2
    cols   = image.shape[1]
    clHf   = cols / 2
    cntr2d = np.zeros( 2 )
    Xlen   = np.tan( np.radians( env_var("_D405_FOV_H_DEG")/2.0 ) ) * zLen
    Ylen   = np.tan( np.radians( env_var("_D405_FOV_V_DEG")/2.0 ) ) * zLen 
    cntr2d = np.array([ ((bbox[0]+bbox[2])/2.0-clHf)/clHf, ((bbox[1]+bbox[3])/2.0-rwHf)/rwHf, ])
    
    return np.array([ cntr2d[0]*Xlen, cntr2d[1]*Ylen, zLen, ])


def KL_div_info_gain_prior_to_post( priorB, postB ):
    """ Get the discrete KL-Divergence of the posterior from the prior """
    rtnKL = 0.0
    for i in range( len( priorB ) ):
        rtnKL += postB[i] * log( postB[i] / priorB[i] )
    return rtnKL


def KL_div_info_gain_dct( priorB : dict, postB : dict ):
    """ Get the discrete KL-Divergence of the posterior from the prior """
    # NOTE: This will throw a `KeyError` if `postB` does not contain at least every key `priorB` does!
    rtnKL = 0.0
    for k in priorB.keys():
        rtnKL += postB[k] * log( postB[k] / priorB[k] )
    return rtnKL


def get_uniform_prior_over_labels( labelsLst : list = None ):
    """ Return a discrete distribution with uniform confusion between classes other than `label` """
    if labelsLst is None:
        labelsLst = env_var("_BLOCK_NAMES") # Defer name fetch until env init
    rtnLabels = {}
    Nclass    = len( labelsLst )
    perProb   = 1.0 / Nclass
    for i in range( Nclass ):
        blkName_i = labelsLst[i]
        rtnLabels[ blkName_i ] = perProb
    return rtnLabels



########## SENSORY PLANNING ########################################################################


class SensoryPlanner:
    """ Do sensing in a way that gets the task done """

    def __init__( self, robot : UR5_Interface, perc : Perception_OWLv2 ):
        """ HACK: THIS IS NOT MEASURED """
        self.robot     = robot
        self.perc      = perc
        self.ZTableCam = -0.081666 - 0.017
        self.dShot     = 3.00*env_var( "_MIN_CAM_PCD_DIST_M" )
        self.dLoc      = 1.25*env_var( "_MIN_CAM_PCD_DIST_M" )


    def tcp_from_cam_pose( self, camPose : np.ndarray ):
        """ Get a robot pose from the camera pose """
        return camPose.dot( np.linalg.inv( np.array( self.robot.camXform ) ) )


    def get_camera_Z_offset( self ):
        """ Bump everything up by some Z value I guess """
        return -self.ZTableCam 


    def plan_3d_shot_centroid( self, objects : list[GraspObj], backupDir : np.ndarray, dBackup : float, defaultPose : np.ndarray ):
        """ Plan a camera pose for along a line to the centroid of the objects """
        rtnPose = defaultPose.copy()

        if len( objects ):
            centroid = np.zeros( 3 )
            for obj in objects:
                centroid += extract_pose_as_homog( obj )[0:3,3].reshape( 3 )
            centroid /= len( objects )
        else:
            centroid = defaultPose[0:3,3].reshape(3)

        backupDr = vec_unit( backupDir ) # vec_unit( [1.0,0.25,1.0] )
        backupVc = backupDr * dBackup
        backupPt = centroid + backupVc
        xBasis = np.array([0.0, -1.0, 0.0])
        zBasis = -backupDr
        yBasis = vec_unit( np.cross( zBasis, xBasis ) )
        xBasis = vec_unit( np.cross( yBasis, zBasis ) )
        rtnPose[0:3,0] = xBasis
        rtnPose[0:3,1] = yBasis
        rtnPose[0:3,2] = zBasis
        rtnPose[0:3,3] = backupPt
        # return self.tcp_from_cam_pose( repair_pose( rtnPose ) )
        return self.tcp_from_cam_pose( rtnPose )
    

    def plan_3d_shots( self, objects : list[GraspObj], defaultPose : np.ndarray ):
        """ A Series of shots  """
        return [
            self.plan_3d_shot_centroid( objects, [  0.75, -0.25, 1.0, ], self.dShot, defaultPose ),
            self.plan_3d_shot_centroid( objects, [  1.00,  0.25, 1.0, ], self.dShot, defaultPose ),
            # self.plan_3d_shot_centroid( objects, [ -1.25,  0.25, 1.0, ], self.dShot, defaultPose ), 
            # self.plan_3d_shot_centroid( objects, [ -1.25, -0.25, 1.0, ], self.dShot, defaultPose ), 
        ]
    

    def locate( self, obj : GraspObj ):
        """ Home in on a partcular block """
        initShot = self.plan_3d_shot_centroid( list(), [0.0, 0.0, 1.0,], self.dLoc, extract_pose_as_homog( obj ) )
        self.robot.moveL( initShot, asynch = False )
        query   = _REVERSE_QUERIES[ obj.label ]['query']
        abbrevq = _REVERSE_QUERIES[ obj.label ]['abbrv']
        
        res = self.perc.bound( query, abbrevq )
        while not len( res['hits'] ):
            res = self.perc.bound( query, abbrevq )

        # 2025-04-22: One-Shot Version
        dMin = 1e9
        for hit in res['hits']:
            offset_i  = image_offset( res['image'], hit['bboxi'], self.dLoc )
            dist_i    = np.linalg.norm( offset_i[:2] )
            if dist_i < dMin:
                offset = offset_i
                dMin   = dist_i
        xyDist = np.linalg.norm( offset[:2] )
        if xyDist > 1.5*env_var("_BLOCK_SCALE"):
            return None

        camPose = self.robot.get_cam_pose()
        tcpOfst = np.dot( camPose[0:3,0:3], offset ).reshape(3)
        print( tcpOfst )
        obj.pose.pose[0:2,3] += tcpOfst[0:2]


    def locate_all( self, objLst : list[GraspObj] ):
        """ Locate one object at a time """
        locLst = objLst[:]
        for i, obj_i in enumerate( objLst ):
            for j, obj_j in enumerate( objLst ):
                if i != j:
                    posn_i = extract_pose_as_homog( obj_i )[0:3,3].reshape(3)
                    posn_j = extract_pose_as_homog( obj_j )[0:3,3].reshape(3)
                    vec_ij = vec_unit( posn_j - posn_i )
                    if vec_ij[2] > 0.0:
                        if np.arctan2( np.linalg.norm( vec_ij[0:2] ), vec_ij[2] ) < np.pi/3.0:
                            try:
                                locLst.remove( obj_i )
                            except ValueError:
                                pass
        for obj in locLst:
            self.locate( obj )



########## POSE CHEATER ############################################################################

class PoseCheater:
    """ Fudge the `Memory` such that things are where they should be """

    def __init__( self, basePose = None, startSymbols = None, fix_labels = False, fix_poses = True ):
        """ Setup local memory """
        self.fixLabel = fix_labels
        self.fixPose  = fix_poses
        self.symbols = deque( [startSymbols,] ) if isinstance( startSymbols, list ) else deque()
        self.base    = extract_pose_as_homog( basePose ) if (basePose is not None) else np.eye(4)


    def log_symbols( self, symLst ):
        """ Store the most recent symbols """
        self.symbols.append( deep_copy_memory_list( symLst ) )


    def log_successful_action( self, poseBgn, poseEnd ):
        """ Move the symbol to where the robot moved it """
        print( f"Moved block by {euclidean_distance_between_symbols( poseBgn, poseEnd )}" )
        lastFrame = deep_copy_memory_list( self.symbols[-1] )
        if len(lastFrame ):
            dMin = 1e9
            sCls : GraspObj = None
            for sym in lastFrame:
                d = euclidean_distance_between_symbols( sym, poseBgn )
                if d < dMin:
                    dMin = d
                    sCls = sym
            sCls.pose = ObjPose( poseEnd )
            self.symbols.append( lastFrame[:] )
        


    def log_failed_action( self, poseBgn, poseEnd ):
        """ We done goofed, Erase symbol """
        # lastFrame = deep_copy_memory_list( self.symbols[-1] )
        # dMin = 1e9
        # sCls : GraspObj = None
        # for sym in lastFrame:
        #     d = euclidean_distance_between_symbols( sym, poseBgn )
        #     if d < dMin:
        #         dMin = d
        #         sCls = sym
        # self.symbols.append( [sym for sym in lastFrame if id(sym) != id(sCls)] )
        self.symbols.append( list() )

        print( f"Could NOT move block by {euclidean_distance_between_symbols( poseBgn, poseEnd )}" )


    # def repair_symbol_poses( self, symLst : list[GraspObj], maxDiff = None ):
    def repair_symbol_poses( self, symLst : list[GraspObj], maxDiff = None ) -> list[GraspObj]:
        """ Adjust the positions of symbols to their last """
        lastFrame : list[GraspObj] = self.symbols[-1]
        rtnSym = list()
        lSet = set([])
        cSet = set([])
        dlta = False

        if self.fixLabel and self.fixPose:
            for j, lSym in enumerate( lastFrame ):
                lSet.add( lSym.label )
                rtnSym.append( lSym )
            for i, rSym in enumerate( symLst ):
                if rSym.label not in lSet:
                    lSet.add( rSym.label )
                    rtnSym.append( rSym )
                    dlta = True

        elif self.fixPose:
            if maxDiff is None:
                maxDiff = 4.0 * env_var('_BLOCK_SCALE')

            print( "CHEAT OBJECTS:" )
            for j, lSym in enumerate( lastFrame ):
                print( f"\t{lSym}" )


            for i, rSym in enumerate( symLst ):
                sMin = None
                dMin = 1e9
                for j, lSym in enumerate( lastFrame ):
                    d_ij = euclidean_distance_between_symbols( rSym, lSym )
                    if (d_ij <= maxDiff) and (d_ij < dMin):
                        dMin = d_ij
                        if id( lSym ) not in lSet:
                            sMin = lSym
                if sMin is not None:
                    lSet.add( id( lSym ) )
                    cSet.add( rSym.label )
                    rSym.pose = sMin.pose
                    dlta = True
                rtnSym.append( rSym )

            for j, lSym in enumerate( lastFrame ):
                if (lSym.label not in cSet):
                    cSet.add( lSym.label )
                    rtnSym.append( lSym )

                # if (id( lSym ) not in lSet) and (lSym.label not in cSet):
                #     collide = False
                #     for i, rSym in enumerate( rtnSym ):
                #         if euclidean_distance_between_symbols( lSym, rSym ) < env_var("_BLOCK_SCALE"):
                #             collide = True
                #             break
                #     if not collide:
                #         rtnSym.append( lSym )
                #         lSet.add( id( lSym ) )
                #         cSet.add( lSym.label )
                
        if dlta:
            self.symbols.append( rtnSym )
        return rtnSym
        



########## OBJECT MEMORY ###########################################################################

##### KL Diverfgence ######################################################

def gen_id():
    """ Get a unique ID as a string """
    return str( uuid4() )


class KLD_Tracker:
    """ Track KL-Divergence as the task evolves """
    def __init__( self ):
        self.history = deque()


    def get_symbol_dict( self, sym : GraspObj ):
        """ Render the symbol as a `dict` with relevant info """
        distLast = get_uniform_prior_over_labels( list( sym.labels.keys() ) )
        if len( self.history ):
            for entry in self.history[-1]:
                if entry['label'] == sym.label:
                    distLast = entry['labels']
                    break
        
        return {
            't'    : now(),
            'id'   : gen_id(),
            'label': sym.label,
            'dist' : deepcopy( sym.labels ),
            'pose' : extract_pose_as_homog( sym ),
            'KLD'  : KL_div_info_gain_dct( distLast, sym.labels ),
            'visit': False,
        }
    

    def record_symbols( self, symbols : list[GraspObj] ):
        """ Create a record of all the symbols at this time """
        frame = list()
        for sym in symbols:
            frame.append( self.get_symbol_dict( sym ) )
        self.history.append( frame )


    def get_label_KL_history( self, qLabel ):
        """ Get the KL-Divergence across all timesteps in the episode so far """
        klHist = list()
        for frame in self.history:
            for sym in frame:
                if sym['label'] == qLabel:
                    klHist.append( sym['KLD'] )
                    break
        return klHist
    

    def get_entry( self, qIndex, qLabel ):
        """ Get the symbol record at `qIndex` and `qLabel`, or return `None` """
        for entry in self.history[ qIndex ]:
            if entry['label'] == qLabel:
                return deepcopy( entry )
        return None
        

    def check_KL_criteria( self, N_falling : int, qLabel : str ):
        """ Return `False` if evidence is gathering for a contrary indication, Otherwise return `True` """
        klHist = self.get_label_KL_history( qLabel )
        if len( self.history ) < N_falling:
            return True
        for i in range( -N_falling, -1 ):
            if (klHist[i] < klHist[i+1]):
                return True
        labelDist = zip_dict_sorted_by_decreasing_value( self.get_entry( -1, qLabel )['dist'] )
        print( labelDist[0][0], "-vs-", qLabel )
        if labelDist[0][0] != qLabel:
            return False
        else:
            return True



##### Object Location & Tracking ##########################################

class Memory:
    """ Object Memory """

    ##### KL-Divergence Tracking #################

    def reset_memory( self ):
        """ Erase memory components """
        self.scan : list[GraspObj] = list()
        self.mult : bool           = False
        self.bMem : BayesMemory    = BayesMemory()
        self.klTr : KLD_Tracker    = KLD_Tracker()
        print( "`Memory` initialized ..." )
        
        
    def plot_KL_history_for_all_obj( self ):
        """ Simple plot of the KL divergence for each symbol """
        labels = [entry['label'] for entry in self.klTr.history[0]]
        klHist = dict()
        for lbl in labels:
            klHist[ lbl ] = self.klTr.get_label_KL_history( lbl )
            print( f"Item {lbl}: Dist = {self.klTr.get_entry( -1, lbl )['dist']}\nKL History: {klHist[ lbl ]}\n" )
        print()

        print( f"About to draw graph of {len( labels )} symbols ..." )
        # Adding the legend
        plt.legend()

        # Adding title and labels
        plt.title('KL Divergence -vs- Time for Required Objects')
        plt.xlabel('Time')
        plt.ylabel('KL Divergence')

        # Display / Render
        if 0:
            plt.show()
        else:
            plt.savefig( f"data/KL-Plot_{now()}.pdf" )

        print( "Graph COMPLETE!" )


    def update_symbol_history( self, symLst ):
        """ Write symbols and their KL-Div to history for tracking """
        self.klTr.record_symbols( symLst )


    ##### Begin / End ############################

    def __init__( self, robot, perc, suppressRecord = False ):
        """ Set up for logging and tracking """
        self.record  = not bool( suppressRecord )
        if self.record:
            self.history = LogPickler( prefix = "EROM-Memories", outDir = "data" )
        else:
            self.history = None
        self.camPlan = SensoryPlanner( robot, perc )
        self.reset_memory()


    def shutdown( self ):
        """ Save the memory """
        # WARNING: LARGE FILE! > 1Gb
        if self.record:
            self.history.dump_to_file( openNext = False )


    def __del__( self ):
        """ Write record on shutdown """
        self.shutdown()


    ##### Perception #############################

    def plan_3d_shots( self, defaultPose : np.ndarray ):
        """ Ask the sensory planner to get us a shot """
        # return self.camPlan.plan_3d_shots( self.scan, defaultPose )
        return self.camPlan.plan_3d_shots( list(), defaultPose )
    

    def locate_all( self, objLst : list[GraspObj] ):
        """ Locate one object at a time """
        self.camPlan.locate_all( objLst )


    def process_observations( self, obs, xform = None, Append = False ):
        """ Integrate one noisy scan into the current beliefs """
        gObs = list()
        if len( obs ):
            if isinstance( obs[0], GraspObj ):
                gObs = obs[:]
            else:
                gObs = observation_to_readings( obs, xform )
            
        if (Append and self.mult):
            self.scan.extend( gObs )
        else:
            self.scan = gObs[:]
            if Append:
                self.mult = True

        rtnBad = self.bMem.belief_update( gObs, xform, maxRadius = env_var("_BAYES_RAD_L2_M") )

        if self.record:
            self.history.append( 
                datum = {
                    "scan"   : deep_copy_memory_list( self.scan ),
                    "beliefs": deep_copy_memory_list( self.bMem.beliefs ),
                },
                msg = "memory" 
            )
        
        return rtnBad
    

    ##### Symbol Grounding #######################


    def get_current_most_likely( self, strat : str | list = "combo" ) -> list[GraspObj]:
        """ Generate symbols """
        symbols = list()

        if (strat == "combo") or isinstance( strat, list ):

            # # HACK: USE POINT COUNT AS A SCALE OF CONFIDENCE
            # self.bMem.scale_by_pcd_pop()

            # symbols = most_likely_objects( self.bMem.beliefs, "unique" )

            if isinstance( strat, list ):
                symbols = most_likely_objects( self.bMem.beliefs, strat )
            else:
                symbols = most_likely_objects( self.bMem.beliefs, "sufficient" )

        else:
            raise ValueError( f"The update strategy {str(strat).upper()} is NOT recognized!" )

        if self.record:
            self.history.append( 
                datum = deep_copy_memory_list( symbols ),
                msg   = "symbols" 
            )

        # # HACK: SNAP SYMBOL Z
        # for sym in symbols:
        #     sym.pose.pose[2,3] = snap_z_to_nearest_block_unit_above_zero( sym.pose.pose[2,3] )

        self.update_symbol_history( symbols )

        return symbols
        