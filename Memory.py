########## INIT ####################################################################################

import time
now = time.time
from random import choice
from collections import deque
from typing import Dict, Deque
from math import log
from uuid import uuid4

import numpy as np

from magpie_control.poses import vec_unit, repair_pose
from magpie_control.ur5 import UR5_Interface

from aspire.env_config import env_var
from aspire.utils import match_name, normalize_dist
from aspire.symbols import ( ObjPose, GraspObj, extract_pose_as_homog, euclidean_distance_between_symbols )


### Local ###
from utils import ( snap_z_to_nearest_block_unit_above_zero, LogPickler, zip_dict_sorted_by_decreasing_value, 
                    deep_copy_memory_list, closest_ray_points )
from OWLv2_Segment import Perception_OWLv2
from Bayes import BayesMemory


_REVERSE_QUERIES = {
    "bluBlock": {'query': "a photo of a blue block"  , 'abbrv': "blu", },
    "ylwBlock": {'query': "a photo of a yellow block", 'abbrv': "ylw", },
    "grnBlock": {'query': "a photo of a green block" , 'abbrv': "grn", },
}



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

    height   = 0.5*env_var("_BLOCK_SCALE")+env_var("_Z_TABLE")

    hackMap  = [ [[minX, minY, height], [ 1.0/100.0,  0.0/100.0, 0.0]],
                 [[minX, maxY, height], [ 1.0/100.0, -1.0/100.0, 0.0]],

                 [[midX, minY, height], [ 3.0/100.0,  0.0/100.0, 0.0]],
                 [[midX, midY, height], [ 2.0/100.0,  0.0/100.0, 0.0]],

                 [[maxX, minY, height], [ 2.0/100.0,  0.0/100.0, 0.0]], 
                 [[maxX, maxY, height], [ 2.0/100.0, -1.0/100.0, 0.0]],]
    
    weights = list()
    for hack in hackMap:
        weights.append( 1.0 / np.linalg.norm( np.subtract( vec, hack[0] ) ) )
    tot = sum( weights )
    for i, hack in enumerate( hackMap ):
        offset += (weights[i]/tot) * np.array( hack[1] )
    hackXfrm[0:3,3] = offset
    return hackXfrm


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
            # HACK: THERE IS A PERSISTENT GRASP OFFSET IN THE SCENE
            if 0:
                hackXfrm = hacked_offset_map( xform.dot( np.array( item['Pose'] ).reshape( (4,4,) ) )  )
                xform    = hackXfrm.dot( xform ) #env_var("_HACKED_OFFSET").dot( xform )
                objPose  = xform.dot( np.array( item['Pose'] ).reshape( (4,4,) ) ) 
            else:
                objPose = xform.dot( np.array( item['Pose'] ).reshape( (4,4,) ) ) 
            
            # HACK: SNAP TO NEAREST BLOCK UNIT && SNAP ABOVE TABLE
            objPose[2,3] = snap_z_to_nearest_block_unit_above_zero( objPose[2,3] + zOffset )
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
            if euclidean_distance_between_symbols( obj_i, obj_j ) < env_var( "_BLOCK_SCALE" ):
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

    # HACK: SNAP TO NEAREST BLOCK UNIT && SNAP ABOVE TABLE
    # for sym in symbols:
    #     # sym.pose.pose[2,3] = snap_z_to_nearest_block_unit_above_zero( sym.pose.pose[2,3] + zOffset )
    #     sym.pose.pose[2,3] = snap_z_to_nearest_block_unit_above_zero( sym.pose.pose[2,3] )

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
            self.plan_3d_shot_centroid( objects, [ -1.25,  0.25, 1.0, ], self.dShot, defaultPose ), 
            self.plan_3d_shot_centroid( objects, [ -1.25, -0.25, 1.0, ], self.dShot, defaultPose ), 
        ]
    

    def locate( self, obj : GraspObj ):
        """ Home in on a partcular block """
        initShot = self.plan_3d_shot_centroid( list(), [0.0, 0.0, 1.0,], self.dLoc, extract_pose_as_homog( obj ) )
        self.robot.moveL( initShot, asynch = False )
        query   = _REVERSE_QUERIES[ obj.label ]['query']
        abbrevq = _REVERSE_QUERIES[ obj.label ]['abbrv']
        res     = self.perc.bound( query, abbrevq )
        while not len( res['hits'] ):
            res = self.perc.bound( query, abbrevq )
        offset  = image_offset( res['image'], res['hits'][0]['bboxi'], self.dLoc )
        curPose = self.robot.get_tcp_pose()
        camPose = self.robot.get_cam_pose()
        while( np.linalg.norm( offset[:2] ) > 0.5*env_var("_PLACE_XY_ACCEPT") ):
            tcpOfst = np.dot( camPose[0:3,0:3], offset ).reshape(3)
            print( tcpOfst )
            if np.linalg.norm( offset[:2] ) > 0.1:
                break
            movPose = curPose.copy()
            movPose[0:2,3] += tcpOfst[0:2]
            obj.pose.pose[0:2,3] += tcpOfst[0:2]
            self.robot.moveL( movPose, asynch = False )
            res = self.perc.bound( query, abbrevq )
            while not len( res['hits'] ):
                res = self.perc.bound( query, abbrevq )
            offset  = image_offset( res['image'], res['hits'][0]['bboxi'], self.dLoc )
            curPose = self.robot.get_tcp_pose()
            camPose = self.robot.get_cam_pose()


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
        self.pose  = np.eye(4) if (pose is None) else pose
        self.distH = list() # Distribution history
        self.KLDvH = list() # KL-Divergence history


##### Object Location & Tracking ##########################################

class Memory:
    """ Object Memory """

    ##### KL-Divergence Tracking #################

    def reset_memory( self ):
        """ Erase memory components """
        self.scan : list[GraspObj]   = list()
        self.mult : bool             = False
        self.bMem : BayesMemory      = BayesMemory()
        self.symH : Dict[ThinSymbol] = dict()
        # self.syHs : list[dict]     = list() # NOT THE WAY TO DO IT!
        # self.klHs : list[float]    = list()


    def closest_symbol_to_pose( self, pose, margin = None ):
        """ Fetch the closest symbol to the pose within `margin`, otherwise return None """
        # FIXME
        pass


    def move_symbol_from_to_pose( self, srcPose, dstPose ):
        """ Find the symbol at `srcPose` and move it to `dstPose` """
        # FIXME
        pass


    def update_symbol_history( self, symLst : list[GraspObj] ):
        """ Find the symbol at `srcPose` and move it to `dstPose` """
        # FIXME
        pass


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
        # HACK: WORKING FROM SCAN, NOT THE BELIEF
        return self.camPlan.plan_3d_shots( self.scan, defaultPose )
    

    def locate_all( self, objLst : list[GraspObj] ):
        """ Locate one object at a time """
        self.camPlan.locate_all( objLst )


    def process_observations( self, obs, xform = None, Append = False ):
        """ Integrate one noisy scan into the current beliefs """
        if (Append and self.mult):
            # HACK: BUMP EVERYTHING UP BY SOME OFFSET
            self.scan.extend( observation_to_readings( obs, xform, self.camPlan.get_camera_Z_offset() ) )
        else:
            # HACK: BUMP EVERYTHING UP BY SOME OFFSET
            self.scan = observation_to_readings( obs, xform, self.camPlan.get_camera_Z_offset() )
            if Append:
                self.mult = True

        self.bMem.belief_update( self.scan, xform )

        self.history.append( 
            datum = {
                "scan"   : deep_copy_memory_list( self.scan ),
                "beliefs": deep_copy_memory_list( self.bMem.beliefs ),
            },
            msg = "memory" 
        )


    def HACK_MERGE( self ):
        """ HACK: Just average the poses """

        rayFac = 5.5 # 3.0 # 8.0

        def ray_merge( objLst : list[GraspObj] ):
            """ What is the mutually closes point between all cam rays? """
            N      = len( objLst )
            pntLst = list()
            for i in range( N-1 ):
                obj_i = objLst[i]
                for j in range( i+1, N ):
                    obj_j = objLst[j]
                    pnt_ij, pnt_ji = closest_ray_points( 
                        obj_i.meta['rayOrg'], 
                        obj_i.meta['rayDir'], 
                        obj_j.meta['rayOrg'], 
                        obj_j.meta['rayDir'], 
                    )
                    pntLst.extend([pnt_ij, pnt_ji,])
            return np.mean( pntLst, axis = 0 )
                    
        cat    = dict()
        rtnLst = list()
        for obj in self.scan:
            labelDist = zip_dict_sorted_by_decreasing_value( obj.labels )
            labelMax  = labelDist[0][0]
            if labelMax in cat:
                cat[ labelMax ].append( obj )
            else:
                cat[ labelMax ] = [ obj, ]
        for k, v in cat.items():
            cntr = np.zeros( 3 )
            for obj_i in v:
                cntr += extract_pose_as_homog( obj_i )[0:3,3].reshape( 3 )
            ryCn = ray_merge( v )
            cntr += ryCn * rayFac
            cntr /= (len(v)+rayFac)
            
            rtnObj = v[0]
            rtnObj.pose.pose[0:3,3] = cntr
            # rtnObj.pose.pose[0:3,3] = ryCn
            rtnObj.pose.pose[2,3] = max( rtnObj.pose.pose[2,3], 0.5*env_var("_BLOCK_SCALE") )
            print( f"There are {len(v)} examples of {k}, Pose:\n{rtnObj.pose.pose[0:3,3]}" )
            rtnLst.append( rtnObj )
        return rtnLst
    

    ##### Symbol Grounding #######################

    def p_KL_OK_per_class( self ):
        """ Evaluate the KL """
        pass


    def get_current_most_likely( self, strat = "bayes" ) -> list[GraspObj]:
        """ Generate symbols """
        symbols = list()
        if strat == "bayes":
            symbols = most_likely_non_conflict( self.bMem.beliefs, self.camPlan.get_camera_Z_offset() ) 
        elif strat == "hack":
            symbols = strongest_symbols_from_readings( self.HACK_MERGE(), env_var("_N_REQD_OBJS") )
        elif strat == "score":
            symbols = strongest_symbols_from_readings( self.scan, env_var("_N_REQD_OBJS") )
        else:
            raise ValueError( f"The update strategy {str(strat).upper()} is NOT recognized!" )

        self.history.append( 
            datum = deep_copy_memory_list( symbols ),
            msg   = "symbols" 
        )

        self.append_to_history( symbols )

        return symbols
        


    
    
    
    