########## INIT ####################################################################################

##### Imports #####

### Standard ###
import time
now = time.time
from collections import deque
from typing import Dict, Deque
from math import log
from uuid import uuid4

### Special ###
import numpy as np

### Local ###
from magpie_control.poses import vec_unit, translation_diff
from magpie_control.ur5 import UR5_Interface

from aspire.env_config import env_var
from aspire.utils import match_name, normalize_dist
from aspire.symbols import ( ObjPose, GraspObj, extract_pose_as_homog, euclidean_distance_between_symbols )

from utils import ( LogPickler, zip_dict_sorted_by_decreasing_value, deep_copy_memory_list, )
from OWLv2_Segment import Perception_OWLv2
from Bayes import BayesMemory


##### Constants #####
_REVERSE_QUERIES = {
    "bluBlock": {'query': "a photo of a blue block"  , 'abbrv': "blu", },
    "ylwBlock": {'query': "a photo of a yellow block", 'abbrv': "ylw", },
    "grnBlock": {'query': "a photo of a green block" , 'abbrv': "grn", },
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
        else:
            raise ValueError( f"`observation_to_readings`: BAD POSE FORMAT!\n{item['Pose']}" )
        
        # item['CPCD']

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


def strongest_symbols_from_readings( objLst : list[GraspObj], N : int ):
    """ Randomly pick `N` readings to serve as symbols """
    if len( objLst ) < N:
        return list()
    
    picked = dict()

    for obj in objLst:
        print( obj.score )
        obj.score = np.mean( obj.score )
        labelDist = zip_dict_sorted_by_decreasing_value( obj.labels )
        
        for lbl_i, prb_i in labelDist:
            if (lbl_i not in picked) or (prb_i > picked[ lbl_i ].prob):
                nu = obj.copy_child()
                nu.label = lbl_i
                nu.prob  = prb_i
                picked[ lbl_i ] = nu
                break
        
    return list( picked.values() )


from pprint import pprint
from copy import deepcopy


def most_likely_non_conflict( objLst : list[GraspObj], zOffset : float ) -> list[GraspObj]:
    """ Choose the most likely in each class that does not conflict with an even more likely label of a different class """

    print( f"There are {len(objLst)} to evaluate!" )
    
    ranked : Dict[str, Deque[GraspObj]] = dict()
    for obj in objLst:
        labelDist = zip_dict_sorted_by_decreasing_value( obj.labels )
        for lbl_i, prb_i in labelDist:
            if (lbl_i not in ranked):
                ranked[ lbl_i ] = deque()
            obj_i = obj.copy_child()
            obj_i.label = lbl_i
            obj_i.prob  = prb_i
            ranked[ lbl_i ].append( obj_i )

    for k, v in ranked.items():
        nuV = list(v)
        nuV.sort( key = lambda x: x.prob, reverse = True )
        ranked[k] = deque( nuV )

    print( "Label Ranking" )
    pprint( ranked )

    picked  : Dict[str, GraspObj] = dict()
    compare : Dict[str, GraspObj] = dict()
    for lbl_i in ranked.keys():
        if lbl_i == env_var( "_NULL_NAME" ):
            continue
        obj_i   = ranked[ lbl_i ].popleft()
        collide = False
        for lbl_j, obj_j in compare.items():
            if euclidean_distance_between_symbols( obj_i, obj_j ) < env_var( "_WIDE_COLLIDE" ):
                collide = True
                if obj_i.prob > obj_j.prob:
                    picked[ lbl_i ] = obj_i
                    picked[ lbl_j ] = ranked[ lbl_j ].popleft()
                    while euclidean_distance_between_symbols( picked[ lbl_i ], picked[ lbl_j ] ) < env_var( "_BLOCK_SCALE" ):
                        if len( ranked[ lbl_j ] ):
                            picked[ lbl_j ] = ranked[ lbl_j ].popleft()
                        else:
                            break
                else:
                    picked[ lbl_i ] = ranked[ lbl_i ].popleft()
                    picked[ lbl_j ] = obj_j
                    while euclidean_distance_between_symbols( picked[ lbl_i ], picked[ lbl_j ] ) < env_var( "_BLOCK_SCALE" ):
                        if len( ranked[ lbl_i ] ):
                            picked[ lbl_i ] = ranked[ lbl_i ].popleft()
                        else:
                            break
        if not collide:
            picked[ lbl_i ] = obj_i
        compare = deepcopy( picked )

    symbols = list( picked.values() )

    print( f"About to return {len(symbols)} symbols!" )
    return symbols
        

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
        self.dShot     = 1.5*env_var( "_MIN_CAM_PCD_DIST_M" )
        self.dLoc      = 1.1*env_var( "_MIN_CAM_PCD_DIST_M" )


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
            self.plan_3d_shot_centroid( objects, [  1.25, -0.25, 1.0, ], self.dShot, defaultPose ),
            self.plan_3d_shot_centroid( objects, [  1.25,  0.25, 1.0, ], self.dShot, defaultPose ),
            # self.plan_3d_shot_centroid( objects, [ -1.25,  0.25, 1.0, ], self.dShot, defaultPose ), 
            self.plan_3d_shot_centroid( objects, [ -1.25, -0.25, 1.0, ], self.dShot, defaultPose ), 
        ]
    

    def locate( self, obj : GraspObj ):
        """ Home in on a partcular block """
        initShot = self.plan_3d_shot_centroid( list(), [0.0, 0.0, 1.0,], self.dLoc, extract_pose_as_homog( obj ) )
        self.robot.moveL( initShot, asynch = False )
        query   = _REVERSE_QUERIES[ obj.label ]['query']
        abbrevq = _REVERSE_QUERIES[ obj.label ]['abbrv']
        
        while( True ):

            res = self.perc.bound( query, abbrevq )
            while not len( res['hits'] ):
                res = self.perc.bound( query, abbrevq )

            if 0:
                dLim = 2*env_var("_BLOCK_SCALE")
                for hit in res['hits']:
                    offset_i  = image_offset( res['image'], hit['bboxi'], self.dLoc )
                    dist_i    = np.linalg.norm( offset_i[:2] )
                    if dist_i < dLim:
                        offset = offset_i
                        break
            else:
                offset = image_offset( res['image'], res['hits'][0]['bboxi'], self.dLoc )

            if np.linalg.norm( offset[:2] ) <= 0.5*env_var("_PLACE_XY_ACCEPT"):
                break

            curPose = self.robot.get_tcp_pose()
            camPose = self.robot.get_cam_pose()

            tcpOfst = np.dot( camPose[0:3,0:3], offset ).reshape(3)
            print( tcpOfst )
            
            movPose = curPose.copy()
            movPose[0:2,3] += tcpOfst[0:2]
            obj.pose.pose[0:2,3] += tcpOfst[0:2]
            self.robot.moveL( movPose, asynch = False )
            

            


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
                        if np.arctan2( np.linalg.norm( vec_ij[0:2] ), vec_ij[2] ) < np.pi/4.0:
                            try:
                                locLst.remove( obj_i )
                            except ValueError:
                                pass
        for obj in locLst:
            self.locate( obj )



########## OBJECT MEMORY ###########################################################################

##### BAD, YAGNI ##########################################################

class ThinSymbol:
    """ Barest Symbol """
    # HACK: IS THIS A BAD THING? YAGNI?
    
    def __init__( self, label = "", pose = None ):
        self.id    = uuid4()
        self.label = label
        self.pose  = np.eye(4) if (pose is None) else extract_pose_as_homog( pose )
        self.distH = list() # Distribution history
        self.KLDvH = list() # KL-Divergence history
        self.visit = False


    def append_dist( self, labelDist : dict ):
        self.distH.append( deepcopy( labelDist ) )
        if len( self.distH ) > 1:
            lstDst = self.distH[-2]
        else:
            lstDst = get_uniform_prior_over_labels( list( labelDist.keys() ) )
        self.KLDvH.append( KL_div_info_gain_dct( lstDst, labelDist ) )


    def check_KL_criteria( self, N_falling : int, expectedLabel : str ):
        """ Return `False` if evidence is gathering for a contrary indication, Otherwise return `True` """
        if len( self.KLDvH ) < N_falling:
            return True
        for i in range( -N_falling, -1 ):
            if (self.KLDvH[i] < self.KLDvH[i+1]):
                return True
        labelDist = zip_dict_sorted_by_decreasing_value( self.distH[-1] )
        print( labelDist[0][0], "-vs-", expectedLabel )
        if labelDist[0][0] != expectedLabel:
            return False
        else:
            return True
        





##### Object Location & Tracking ##########################################

class Memory:
    """ Object Memory """

    ##### KL-Divergence Tracking #################

    def reset_memory( self ):
        """ Erase memory components """
        self.scan : list[GraspObj]   = list()
        self.mult : bool             = False
        self.bMem : BayesMemory      = BayesMemory()
        self.symH : Dict[uuid4,ThinSymbol] = dict()
        # self.syHs : list[dict]     = list() # NOT THE WAY TO DO IT!
        # self.klHs : list[float]    = list()


    def closest_symbol_to_pose( self, pose, margin = None ) -> ThinSymbol:
        """ Fetch the closest symbol to the pose within `margin`, otherwise return None """
        if margin is None:
            margin = 2.0 * env_var("_BLOCK_SCALE")
        pose = extract_pose_as_homog( pose )
        dMin = 1e9
        sMin = None
        for v in self.symH.values():
            d = translation_diff( pose, v.pose )
            if d < dMin:
                dMin = d
                sMin = v
        if dMin <= margin:
            return sMin
        else:
            return None


    def move_symbol_from_to_pose( self, srcPose, dstPose ):
        """ Find the symbol at `srcPose` and move it to `dstPose`, Return thin symbols if it was moved, else return None """
        dstPose  = extract_pose_as_homog( dstPose )
        needMove = self.closest_symbol_to_pose( srcPose )
        if (needMove is not None):
            needMove.pose = dstPose.copy()
            return needMove
        else:
            return None


    def update_symbol_history( self, symLst : list[GraspObj] ):
        """ Match new symbols to current and calculate confidence changes """
        for sym in symLst:
            tSm = self.closest_symbol_to_pose( sym )
            if tSm is None:
                nuS = ThinSymbol( label = sym.label, pose = sym.pose )
                nuS.append_dist( sym.labels )
                self.symH[ nuS.id ] = nuS
            else:
                tSm.append_dist( sym.labels )


    def check_KL_for_symbol_at_pose( self, pose, expectedLabel : str, poseMargin : float = None, N_falling : int = 3 ):
        """ Return `check_KL_criteria` for the symbol nearest this pose """
        chkSym = self.closest_symbol_to_pose( pose, margin = poseMargin )
        if chkSym is not None:
            return chkSym.check_KL_criteria( N_falling, expectedLabel )
        else:
            return False


    ##### Begin / End ############################

    def __init__( self, robot, perc ):
        self.history = LogPickler( prefix = "EROM-Memories", outDir = "data" )
        self.camPlan = SensoryPlanner( robot, perc )
        self.reset_memory()


    def shutdown( self ):
        """ Save the memory """
        self.history.dump_to_file( openNext = False )


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
        if (Append and self.mult):
            self.scan.extend( observation_to_readings( obs, xform ) )
        else:
            self.scan = observation_to_readings( obs, xform )
            if Append:
                self.mult = True

        # self.bMem.belief_update( self.scan, xform )
        self.bMem.belief_update( self.scan, xform, maxRadius = env_var("_BAYES_RAD_L2_M") )

        self.history.append( 
            datum = {
                "scan"   : deep_copy_memory_list( self.scan ),
                "beliefs": deep_copy_memory_list( self.bMem.beliefs ),
            },
            msg = "memory" 
        )
    

    ##### Symbol Grounding #######################


    def get_current_most_likely( self, strat = "bayes" ) -> list[GraspObj]:
        """ Generate symbols """
        symbols = list()
        if strat == "bayes":
            symbols = most_likely_non_conflict( self.bMem.beliefs, self.camPlan.get_camera_Z_offset() ) 
        elif strat == "score":
            symbols = strongest_symbols_from_readings( self.scan, env_var("_N_REQD_OBJS") )
        else:
            raise ValueError( f"The update strategy {str(strat).upper()} is NOT recognized!" )

        self.history.append( 
            datum = deep_copy_memory_list( symbols ),
            msg   = "symbols" 
        )

        self.update_symbol_history( symbols )

        return symbols
        


    
    
    
    