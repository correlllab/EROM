########## INIT ####################################################################################

import time
now = time.time
from random import choice

import numpy as np

from magpie_control.poses import vec_unit, repair_pose
from magpie_control.ur5 import UR5_Interface

from aspire.env_config import env_var
from aspire.utils import match_name, normalize_dist
from aspire.symbols import ( ObjPose, GraspObj, extract_pose_as_homog, )


### Local ###
from utils import ( snap_z_to_nearest_block_unit_above_zero, LogPickler, 
                    zip_dict_sorted_by_decreasing_value, deep_copy_memory_list, )



########## GEOMETRY FUNCTIONS ######################################################################

def closest_ray_points( A_org, A_dir, B_org, B_dir ):
    """ Return the closest point on ray A to ray B and on ray B to ray A """
    # https://palitri.com/vault/stuff/maths/Rays%20closest%20point.pdf
    c  = np.subtract( B_org, A_org )
    aa = np.dot( A_dir, A_dir )
    bb = np.dot( B_dir, B_dir )
    ab = np.dot( A_dir, B_dir )
    ac = np.dot( A_dir, c     )
    bc = np.dot( B_dir, c     ) 
    fA = (-ab*bc + ac*bb) / (aa*bb-ab*ab)
    fB = ( ab*ac - bc*aa) / (aa*bb-ab*ab)
    pA = np.add( A_org, np.multiply( A_dir, fA ) )
    pB = np.add( B_org, np.multiply( B_dir, fB ) )
    return pA, pB



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
        



########## SENSORY PLANNING ########################################################################


class SensoryPlanner:
    """ Do sensing in a way that gets the task done """

    def __init__( self, robot : UR5_Interface ):
        """ HACK: THIS IS NOT MEASURED """
        self.robot     = robot
        self.ZTableCam = -0.081666 - 0.017
        self.dShot     = 1.75*env_var( "_MIN_CAM_PCD_DIST_M" )


    def tcp_from_cam_pose( self, camPose : np.ndarray ):
        """ Get a robot pose from the camera pose """
        return camPose.dot( np.linalg.inv( np.array( self.robot.camXform ) ) )


    def get_camera_Z_offset( self ):
        """ Bump everything up by some Z value I guess """
        return -self.ZTableCam 


    def plan_3d_shot( self, objects : list[GraspObj], backupDir : np.ndarray, defaultPose : np.ndarray ):
        """ Plan a camera pose that maximizes info and avoids occlusion, given the proposed objects """
        rtnPose = defaultPose.copy()
        if len( objects ):
            centroid = np.zeros( 3 )
            for obj in objects:
                centroid += extract_pose_as_homog( obj )[0:3,3].reshape( 3 )
            centroid /= len( objects )
            backupDr = vec_unit( backupDir ) # vec_unit( [1.0,0.25,1.0] )
            backupVc = backupDr * self.dShot
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
        return [
            self.plan_3d_shot( objects, [  1.00, 0.25, 1.0, ], defaultPose ),
            # self.plan_3d_shot( objects, [  1.00,-0.25, 1.0, ], defaultPose ), # Pulled too far +X???
            self.plan_3d_shot( objects, [ -0.25, 0.00, 1.0, ], defaultPose ),
            # self.plan_3d_shot( objects, [ -0.50, 0.00, 1.0, ], defaultPose ), # Try in a bit
        ]




########## OBJECT MEMORY ###########################################################################

class Memory:
    """ Object Memory """

    def reset_memory( self ):
        """ Erase memory components """
        self.scan : list[GraspObj] = list()
        self.mult : bool           = False


    def __init__( self, robot ):
        self.history = LogPickler( prefix = "EROM-Memories", outDir = "data" )
        self.camPlan = SensoryPlanner( robot )
        self.reset_memory()


    def shutdown( self ):
        """ Save the memory """
        self.history.dump_to_file( openNext = False )


    def plan_3d_shots( self, defaultPose : np.ndarray ):
        """ Ask the sensory planner to get us a shot """
        # HACK: WORKING FROM SCAN, NOT THE BELIEF
        return self.camPlan.plan_3d_shots( self.scan, defaultPose )


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

        self.history.append( 
            datum = {
                "scan": deep_copy_memory_list( self.scan ),
            },
            msg = "memory" 
        )


    def HACK_MERGE( self ):
        """ HACK: Just average the poses """

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
            cntr += ryCn
            cntr /= (len(v)+1)
            
            rtnObj = v[0]
            # rtnObj.pose.pose[0:3,3] = cntr
            rtnObj.pose.pose[0:3,3] = ryCn
            print( f"There are {len(v)} examples of {k}, Pose:\n{rtnObj.pose.pose[0:3,3]}" )
            rtnLst.append( rtnObj )
        return rtnLst


    def get_current_most_likely( self ):
        """ Generate symbols """
        if self.mult:
            symbols = strongest_symbols_from_readings( self.HACK_MERGE(), env_var("_N_REQD_OBJS") )
        else:
            symbols = strongest_symbols_from_readings( self.scan, env_var("_N_REQD_OBJS") )

        self.history.append( 
            datum = deep_copy_memory_list( symbols ),
            msg   = "symbols" 
        )

        return symbols
        


    
    
    
    