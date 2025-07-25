from __future__ import annotations

import cmath, math, os
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

from random import random, choice
from collections import deque
from collections.abc import Callable
from copy import deepcopy

import numpy as np
from numpy import linalg
from vispy import scene
from vispy.visuals import transforms

from magpie_control.poses import vec_unit
from magpie_control.ur5 import UR5_Interface
from aspire.symbols import euclidean_distance_between_symbols, GraspObj, extract_pose_as_homog, extract_position
from aspire.env_config import env_var

from homog_utils import ( posn_from_xform, bases_from_xform, R_krot, R_z, homog_xform, diff_mag, apply_homog_to_direction_vec,
                          diff_unit, )
from dh_mp import FK_DH_chain, dh_link_homog

from Geometry import get_D405_FOV_frustum, p_sphere_inside_plane_list, grid_points_on_plane, bases_from_xB_zB
from draw_beliefs import render_memory_list
from utils import get_pose_attr

# from TaskPlanner import TaskPlanner

# _RBT_BASE_BUFFER  = 0.200
_RBT_BASE_BUFFER  = 0.300
# _RBT_BASE_FACTOR  = 1.500
# _RBT_TABLE_MARGIN = 0.070
# _RBT_TABLE_MARGIN = 0.140
_RBT_TABLE_MARGIN = 0.210
_REVERSE_QUERIES  = {
    "bluBlock": {'query': "a photo of a small block", 'abbrv': "blu", },
    "ylwBlock": {'query': "a photo of a small block", 'abbrv': "ylw", },
    "grnBlock": {'query': "a photo of a small block", 'abbrv': "grn", },
    "redBlock": {'query': "a photo of a small block", 'abbrv': "red", },
}

##### Globals #####

global mat
mat=np.matrix


########## HELPER FUNCTIONS ########################################################################

def euclidean_distance_between_poses( pose1, pose2 ):
    """ Return the linear distance between two poses """
    return diff_mag( posn_from_xform( pose1 ), posn_from_xform( pose2 ) )


def diff_unit_from_pose1_to_pose2( pose1, pose2 ):
    """ Return the unit linear direction from `pose1` to `pose2` """
    return diff_unit( posn_from_xform( pose2 ), posn_from_xform( pose1 )  )
    



########## DH PARAMETERS ###########################################################################


global d1, a2, a3, d4, d5, d6
d1 =  0.089159
a2 = -0.425
a3 = -0.39225
d4 =  0.10915
d5 =  0.09465
d6 =  0.0823

global d, a, alph

UR5distMod = 0.10915 # Offset shows Link 2 and it's COM where it belongs in space

d    = mat([0.089159, UR5distMod, -UR5distMod, 0.10915, 0.09465, 0.0823])
a    = mat([0 ,-0.425 ,-0.39225 ,0 ,0 ,0])
alph = mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0 ])

plane = [0,0,0,-0.5] #probably wrong but it still works



########## KINEMATICS ##############################################################################

def AH( n, th, c ):
    """ forward kinematics here -- taken from Ryan Keating code """

    T_a      = mat(np.identity(4), copy=False)
    T_a[0,3] = a[0,n-1]
    T_d      = mat(np.identity(4), copy=False)
    T_d[2,3] = d[0,n-1]

    Rzt = mat( [ [cos(th[n-1,c]), -sin(th[n-1,c]), 0 ,0] ,
                 [sin(th[n-1,c]),  cos(th[n-1,c]), 0, 0] ,
                 [0,               0,              1, 0] ,
                 [0,               0,              0, 1] ] , copy = False )
      

    Rxa = mat( [ [1, 0,                 0,                  0] ,
                 [0, cos(alph[0,n-1]), -sin(alph[0,n-1]),   0] ,
                 [0, sin(alph[0,n-1]),  cos(alph[0,n-1]),   0] ,
                 [0, 0,                 0,                  1] ] , copy = False )

    A_i = T_d * Rzt * T_a * Rxa


    return A_i


def HTrans( th, c, num = 6 ):  
    """ Get the effector transform """
    A_1 = AH( 1, th, c )
    A_2 = AH( 2, th, c )
    A_3 = AH( 3, th, c )
    A_4 = AH( 4, th, c )
    A_5 = AH( 5, th, c )
    A_6 = AH( 6, th, c )
    T_01 = A_1
    T_02 = A_1 * A_2
    T_03 = A_1 * A_2 * A_3
    T_04 = A_1 * A_2 * A_3 * A_4 
    T_05 = A_1 * A_2 * A_3 * A_4 * A_5   
    T_06 = A_1 * A_2 * A_3 * A_4 * A_5 * A_6

    transforms = [T_01, T_02, T_03, T_04, T_05, T_06]

    return transforms[num-1]


# def isJointPosSafe( joints , plane ):
#     """ #a*x+b*y+c*z+d """
    
#     # do fwd kinematics
#     th     = np.matrix( [[joints[0]], [joints[1]], [joints[2]], [joints[3]], [joints[4]], [joints[5]]] )
#     c      = [0]
#     fk     = HTrans(th, c)
#     coords = [t[0] for t in fk[:3,3].tolist()]
    
#     for i in range(1,7):
#         fk     = HTrans(th, c, i)
#         coords = [t[0] for t in fk[:3,3].tolist()]
#         val    = (plane[0] * coords[0]) + (plane[1] * coords[1]) + (plane[2] * coords[2]) + plane[3]
        
#         if val >= 0:
#             return False    
#     return True



########## INVERSE KINEMATICS ######################################################################
    
def invKine( desired_pos ):# T60
    """ #inverse kinematics pasted here
        #KEEP ROWS AND COLUMNS STRAIGHT
        #each column is a result to check
        #8 RESULTS, 6 JOINTS  """
    
    # if len( desired_pos ) == 4:
    #     desired_pos = pose_mtrx_to_vec( desired_pos )

    # print( f"Target:\n{desired_pos}" )

    th   = mat( np.zeros((6, 8)) )
    P_05 = ( desired_pos * mat([0,0, -d6, 1]).T-mat([0,0,0,1 ]).T )

    # **** theta1 ****

    psi = atan2(P_05[2-1,0], P_05[1-1,0])
    # phi = acos(d4 / sqrt(P_05[2-1,0]*P_05[2-1,0] + P_05[1-1,0]*P_05[1-1,0]) )

    try:
        phi = acos(d4 / sqrt(P_05[2-1,0]*P_05[2-1,0] + P_05[1-1,0]*P_05[1-1,0]) )
    except ValueError:
        return None

    #The two solutions for theta1 correspond to the shoulder
    #being either left or right
    th[0, 0:4] = pi/2 + psi + phi
    th[0, 4:8] = pi/2 + psi - phi
    th = th.real

    # **** theta5 ****

    cl = [0, 4]# wrist up or down
    for i in range(0,len(cl)):
        c    = cl[i]
        T_10 = linalg.inv(AH(1,th,c))
        T_16 = T_10 * desired_pos
        th[4, c:c+2  ] = +acos( (T_16[2,3]-d4) / d6 );
        th[4, c+2:c+4] = -acos( (T_16[2,3]-d4) / d6 );

    th = th.real

    # **** theta6 ****
    # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

    cl = [0, 2, 4, 6]
    for i in range(0,len(cl)):
        c    = cl[i]
        T_10 = linalg.inv( AH(1,th,c) )
        T_16 = linalg.inv( T_10 * desired_pos )
        th[5, c:c+2] = atan2( (-T_16[1,2]/sin(th[4, c])),(T_16[0,2]/sin(th[4, c])) )
		  
    th = th.real

    # **** theta3 ****
    cl = [0, 2, 4, 6]
    for i in range(0,len(cl)):
        c    = cl[i]
        T_10 = linalg.inv(AH(1,th,c))
        T_65 = AH( 6,th,c)
        T_54 = AH( 5,th,c)
        T_14 = ( T_10 * desired_pos) * linalg.inv(T_54 * T_65)
        P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0,0,0,1]).T
        t3   = cmath.acos((linalg.norm(P_13)**2 - a2**2 - a3**2 )/(2 * a2 * a3)) # norm ?
        th[2, c  ] = +t3.real
        th[2, c+1] = -t3.real

    # **** theta2 and theta 4 ****

    cl = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(0,len(cl)):
        c    = cl[i]
        T_10 = linalg.inv(AH( 1,th,c ))
        T_65 = linalg.inv(AH( 6,th,c))
        T_54 = linalg.inv(AH( 5,th,c))
        T_14 = (T_10 * desired_pos) * T_65 * T_54
        P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0,0,0,1]).T

        # theta 2
        th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(a3* sin(th[2,c])/linalg.norm(P_13))
        # theta 4
        T_32 = linalg.inv(AH( 3,th,c))
        T_21 = linalg.inv(AH( 2,th,c))
        T_34 = T_32 * T_21 * T_14
        th[3, c] = atan2(T_34[1,0], T_34[0,0])
    th = th.real
    return th



########## HELPER FUNCTIONS ########################################################################

def sample_on_sphere( center = [0.0, 0.0, 0.0,], radius = 1.0, N = 1 ):
    """ Generate `N` point(s) on a sphere with `center` and `radius` """
    center = np.array( center )
    radius = abs( radius )
    N      = int( N )

    def gen_pnt():
        """ Get one point """
        pnt = np.array([ -1.0+2.0*random() for _ in range(3) ])
        mag = np.linalg.norm( pnt )
        if mag > 0.0:
           pnt /= mag
           pnt *= radius
           return pnt + center
        else:
            return np.array([1.0, 0.0, 0.0,])
        
    if N == 1:
        return gen_pnt()
    elif N > 1:
        rtnLst = deque()
        for _ in range(N):
            rtnLst.append( gen_pnt() )
        return np.array( list( rtnLst ) )
    

def sphere_samples_to_eff_poses( testPts, center = [0.0, 0.0, 0.0,], effXbasis = [0.0, -1.0, 0.0] ):
    rtnPoses = deque()
    for pnt in testPts:
        rtnPose  = np.eye(4)
        backupVc = np.subtract( pnt, center )
        backupDr = vec_unit( backupVc ) 
        xBasis   = np.array( effXbasis )
        zBasis   = -backupDr
        yBasis   = vec_unit( np.cross( zBasis, xBasis ) )
        xBasis   = vec_unit( np.cross( yBasis, zBasis ) )
        rtnPose[0:3,0] = xBasis
        rtnPose[0:3,1] = yBasis
        rtnPose[0:3,2] = zBasis
        rtnPose[0:3,3] = pnt
        rtnPoses.append( rtnPose.copy() )
    return list( rtnPoses )


def randf( lo = 0.0, hi = 1.0 ):
    """ Return a random `float` between `lo` and `hi` """
    return lo+(hi-lo)*random()


def vary_eff_wrist_3( effPoses : list[np.ndarray], N = 3, lo = -np.pi/2.0, hi = np.pi/2.0 ):
    """ Spin each effector post about local Z """
    rtnPoses = deque()
    for pose_i in effPoses:
        for _ in range(N):
            T      = homog_xform( rotnMatx = R_z( randf( lo, hi ) ) )
            pose_j = pose_i.dot( T )
            rtnPoses.append( pose_j )
    return list( rtnPoses )


def UR5_Jacobian( q : list | np.ndarray ):
    """ Get the full velocity Jacobian from the config """    
    # Source: https://www.researchgate.net/publication/365895438_Singularity_Analysis_and_Complete_Methods_to_Compute_the_Inverse_Kinematics_for_a_6-DOF_URTM-Type_Robot/figures?lo=1
    c1   = cos( q[0] )
    c234 = cos( q[1] + q[2] + q[3] )
    s234 = sin( q[1] + q[2] + q[3] )
    s5   = sin( q[4] )
    c5   = cos( q[4] )
    s1   = sin( q[0] )
    r13  = -c1*c234*s5 + c5*s1
    r23  = -c234*s1*s5 - c1*c5 
    r33  = -s234*s5
    c23  = cos( q[1] + q[2] )
    s23  = sin( q[1] + q[2] )
    c2   = cos( q[1] )
    s2   = sin( q[1] )
    px   = r13*d6 + c1*(s234*d5 + c23*a3 + c2*a2) + s1*d4 
    py   = r23*d6 + s1*(s234*d5 + c23*a3 + c2*a2) - c1*d4
    pz   = r33*d6 - c234*d5 + s23*a3 + s2*a2 + d1 
    J_A = np.array([
        [0.0,  s1,  s1,  s1,  c1*s234, r13],
        [0.0, -c1, -c1, -c1,  s1*s234, r23],
        [1.0, 0.0, 0.0, 0.0, -c234   , r33],
    ])
    J_L1 = np.array([
        [-py,],
        [ px,],
        [0.0,],
    ])
    J_L2 = np.array([
        [-c1*(pz - d1),],
        [-s1*(pz - d1),],
        [s1*py + c1*px,],
    ])
    J_L3 = np.array([
        [c1*(s234*s5*d6 + c234*d5 - s23*a3),],
        [s1*(s234*s5*d6 + c234*d5 - s23*a3),],
        [-c234*s5*d6 + s234*d5 + c23*a3,],
    ])
    J_L4 = np.array([
        [c1*(s234*s5*d6 + c234*d5),],
        [s1*(s234*s5*d6 + c234*d5),],
        [-c234*s5*d6 + s234*d5,],
    ])
    J_L5 = np.array([
        [-d6*(s1*s5 + c1*c234*c5),],
        [d6*(c1*s5 - c234*c5*s1),],
        [-c5*s234*d6,],
    ])
    J_L6 = np.array([
        [0.0,],
        [0.0,],
        [0.0,],
    ])
    return np.vstack( (np.hstack( (J_L1,J_L2,J_L3,J_L4,J_L5,J_L6,) ), J_A,) )


def UR5_manip_score( q : list | np.ndarray ):
    """ Get the manipulability score of the config """
    return np.linalg.det( UR5_Jacobian( q ) ) # 0.0 is BAD
    

def p_all_joints_above_point_normal_plane( joints , point, normal, margin = _RBT_TABLE_MARGIN ):
    """ Check if all joints are on the positive side of a point-normal plane """
    normal = vec_unit( normal )
    th     = np.matrix( [[joints[0]], [joints[1]], [joints[2]], [joints[3]], [joints[4]], [joints[5]]] )
    c      = [0]
    fk     = HTrans(th, c)
    coords = [t[0] for t in fk[:3,3].tolist()]
    
    for i in range(1,7):
        fk     = HTrans(th, c, i)
        coords = [t[0] for t in fk[:3,3].tolist()]
        diff   = np.subtract( coords, point )
        dotPrd = np.dot( diff, normal )
        
        if (dotPrd-margin) <= 0:
            return False    
    return True


def angle_between_vectors_rad( vec1, vec2 ):
    """ Get the angle between vectors in radians """
    if( diff_mag( vec1, vec2 ) < 0.001 ): 
        return 0.0
    return np.arccos( np.dot( vec1, vec2 ) / ( np.linalg.norm( vec1 ) * np.linalg.norm( vec2 ) ) )


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


def get_aabb( ptsLst ):
    try:
        ptsLst = np.array( ptsLst )
        rtnBB  = np.zeros( (2,ptsLst.shape[1],) )
        ptMin  = ptsLst.min( axis = 0 )
        ptMax  = ptsLst.max( axis = 0 )
        rtnBB[0,:] = ptMin
        rtnBB[1,:] = ptMax
    except ValueError:
        print( f"`get_oPCD_aabb`: Array size error! {ptsLst.shape}" )
    return rtnBB


########## MOTION PLANNER ##########################################################################
_COLLISION_NRG_PENALTY = 11.0
_JOINT_Q_MARGIN        = np.pi/8.0

class LUMP:
    """ [L]imited [U]R5 [M]otion [P]lanner """

    ## Problem-Specific Static Vars ##
    ZTableCam   : float = -0.081666 - 0.017
    dShot       : float = 3.00*env_var( "_MIN_CAM_PCD_DIST_M" )
    dLoc        : float = 1.25*env_var( "_MIN_CAM_PCD_DIST_M" )
    _SEP_DIST_M : float =  0.150

    def __init__( self, qInit = None, robot : UR5_Interface = None ):
        """ Set params """
        ## Intenal Scoring ##
        self.q         : np.ndarray    = np.array( [0.0 for _ in range(6)] )
        self.pose      : np.ndarray    = self.FK( self.q )
        self.robot     : UR5_Interface = robot
        self.obstacles : list          = list()
        self.NshotDflt : int           = 3
        self.nextNshot : int           = self.NshotDflt
        if isinstance( qInit, (list, np.ndarray) ):
            self.q = np.array( qInit )
        self.searchActive = False
        self.shotHist     = deque()
        self.qLimLo       = [-np.pi for _ in range(6)]
        self.qLimHi       = [ np.pi for _ in range(6)]
        self.qLimLo[5]   -= np.pi
        self.qLimHi[5]   += np.pi

    def set_state_from_robot( self ):
        if isinstance( self.robot, UR5_Interface ):
            self.q    = self.robot.get_joint_angles()
            self.pose = self.robot.get_tcp_pose()
        else:
            self.q    = np.array( [0.0 for _ in range(6)] )
            self.pose = self.FK( self.q )


    def log_failed_perc( self ):
        """ Tell planner we didn't get enough info last time """
        self.nextNshot += 1


    def log_success_perc( self ):
        """ Tell planner info from last time was good """
        self.nextNshot = self.NshotDflt


    @staticmethod
    def FK( q : list | np.ndarray ):
        """ Perform forward kinematics """
        th = np.matrix( [[q[0]], [q[1]], [q[2]], [q[3]], [q[4]], [q[5]]] )
        c  = [0]
        return HTrans( th, c )
    

    @staticmethod
    def FK_all( q : list | np.ndarray ):
        """ Perform forward kinematics for all frames """
        th  = np.matrix( [[q[0]], [q[1]], [q[2]], [q[3]], [q[4]], [q[5]]] )
        c   = [0]
        rtn = list()
        for i in range(1,7):
            rtn.append( HTrans( th, c, i ) )
        return rtn


    def p_base_safe( self, effPose : np.ndarray ):
        """ Return true if the effector pose is sufficiently far from the base """
        return (euclidean_distance_between_symbols( np.eye(4), effPose ) >= _RBT_BASE_BUFFER)


    def p_nonneg_Z( self, effPose : np.ndarray, margin = _RBT_TABLE_MARGIN ):
        """ Return True if the Z-position is non-negative """
        return ((effPose[2,3] - margin) >= 0.0)
    

    def p_q_within_limits( self, q, margin = _JOINT_Q_MARGIN ):
        """ Check if `q` is within joint limits, including margin """
        for i, q_i in enumerate(q):
            if (q_i-margin) < self.qLimLo[i]:
                return False
            if (q_i+margin) > self.qLimHi[i]:
                return False
        return True
    

    def register_aabb_obstacle( self, aabb ):
        """ Add an Axis-Aligned Boudning Box that the robot should avoid """
        self.obstacles.append({ 'type': "aabb", 'geo' : np.array( aabb ) })


    @staticmethod
    def p_point_in_aabb( pnt, aabb, margin = _RBT_TABLE_MARGIN ):
        """ Return True if the `pnt` is inside the `aabb` of arbitrary dimension """
        ans = True
        for dim, coord in enumerate( pnt ):
            ans = ans and (aabb[0,dim] < (coord - margin))
            ans = ans and (aabb[1,dim] > (coord + margin))
        return ans
    

    def p_safe_pose( self, effPose : np.ndarray ):
        """ Should the robot even consider this pose? """
        return (self.p_base_safe( effPose ) and self.p_nonneg_Z( effPose ))


    def p_collision_q( self, q, margin = _RBT_TABLE_MARGIN ):
        """ Return true if the `q` would put the robot in collision """
        if not self.p_q_within_limits(q):
            return True
        for obstacle in self.obstacles:
            if obstacle["type"] == "aabb":
                aabb   = obstacle["geo"]
                frames = LUMP.FK_all( q )
                if not self.p_safe_pose( frames[-1] ): # Check base<->effector<->table collision
                    return True
                for frm in frames:
                    posn = extract_position( frm )
                    if LUMP.p_point_in_aabb( posn, aabb, margin ):
                        return True
            else:
                raise ValueError( f"`LUMP.p_collision_q()`, UNDEFINED obstacle:\n{obstacle}\n" )
        return False
    

    @staticmethod
    def config_energy( qRef : list | np.ndarray, q : list | np.ndarray ):
        """ Compute a "joint position badness" """
        mag = np.linalg.norm( np.subtract( q, qRef ) )
        return mag + (1.0-UR5_manip_score( q ))*mag


    def IK_search( self, effPose : np.ndarray ):
        """ Perform inverse kinematics (deterministic) """
        solns = invKine( effPose )
        if solns is None:
            return None
        qFltr = list()
        for c in solns.T:
            arr = c.tolist()[0]
            if p_all_joints_above_point_normal_plane( arr, [0.0,0.0,0.0,], [0.0,0.0,1.0,], margin = 0.070 ):
                qFltr.append( arr )
            else:
                pass
                # print( f"UNSAFE: {arr}" )
        eMin = 1e9
        qMin = None
        for soln in qFltr:
            nrg = self.config_energy( self.q, soln )
            hit = 1.0 if self.p_collision_q( soln ) else 0.0
            nrg += hit*_COLLISION_NRG_PENALTY
            if nrg < eMin:
                eMin = nrg
                qMin = soln
        return qMin
    

    def IK( self, effPose : np.ndarray, suppressCache = False ):
        """ Perform inverse kinematics (conditional) """
        soln = self.IK_search( effPose )
        if ((not suppressCache) and (soln is not None) and self.p_safe_pose( effPose )):
            self.q    = np.array( soln )
            self.pose = np.array( effPose )
        return soln
    

    @staticmethod
    def tcp_from_cam_pose( camPose : np.ndarray, robot : UR5_Interface  ):
        """ Get a robot pose from the camera pose """
        return camPose.dot( np.linalg.inv( np.array( robot.camXform ) ) )


    def get_camera_Z_offset( self ):
        """ Bump everything up by some Z value I guess """
        return -self.ZTableCam 
    

    @staticmethod
    def symbol_centroid( objects : list[GraspObj] ):
        """ Get the position centroid of all the objects """
        centroid = np.zeros( 3 )
        for obj in objects:
            centroid += extract_pose_as_homog( obj )[0:3,3].reshape( 3 )
        centroid /= len( objects )
        return centroid
    

    def plan_3d_shot_centroid( self, objects : list[GraspObj], dBackup : float = dShot, N : int = 16, 
                                     energyFunc : Callable = None ):
        """ Plan a camera pose for along a line to the centroid of the objects """
        if energyFunc is None:
            energyFunc = self.config_energy
        if len( objects ):
            centroid = self.symbol_centroid( objects )
        else:
            return None
        
        sphrPts = sample_on_sphere( centroid, dBackup, N*2 )
        testPts = deque()
        for pnt in sphrPts:
            if pnt[2] > 0.0:
                testPts.append( pnt )
        testPts = list( testPts )
        ranking = deque()

        for pnt in testPts:
            rtnPose  = np.eye(4)
            backupVc = np.subtract( pnt, centroid )
            backupDr = vec_unit( backupVc ) 
            xBasis   = np.array([0.0, -1.0, 0.0])
            # xBasis   = np.array([1.0, 0.0, 0.0]) # 2025-06-30: Does NOT work!
            zBasis   = -backupDr
            yBasis   = vec_unit( np.cross( zBasis, xBasis ) )
            xBasis   = vec_unit( np.cross( yBasis, zBasis ) )
            rtnPose[0:3,0] = xBasis
            rtnPose[0:3,1] = yBasis
            rtnPose[0:3,2] = zBasis
            rtnPose[0:3,3] = pnt
            rtnSoln = self.IK( rtnPose, suppressCache = True )
            # if is_pose_mtrx( rtnPose ):
            #     rtnSoln = self.IK( rtnPose, suppressCache = True )
            # else:
            #     rtnSoln = None
            # rtnSoln = pose_vec_to_mtrx( rtnSoln )
            # print( rtnSoln )
            if ((rtnSoln is not None) and self.p_safe_pose( rtnPose )):
                ranking.append((
                    energyFunc( self.q, rtnSoln ),
                    np.array( rtnSoln ),
                ))
            else:
                pass
                # print( f"Cannot Rank: {rtnSoln}" )

        ranking = list( ranking )
        # ranking.sort( key = lambda x: x[0], reverse = True )
        ranking.sort( key = lambda x: x[0] )

        if len( ranking ):
            return ranking[0][1], self.FK( ranking[0][1] )
        else:
            return None
    

    def make_path_safe( self, bgnPose, endPose ):
        """ If the straight-line path would pass too close to the robot base, then propose an intermediate waypoint """
        _EXTRA_PAD_BASE = 0.400
        bgnPosn = extract_position( bgnPose )
        endPosn = extract_position( endPose )
        trvlDir = vec_unit( np.subtract( endPosn, bgnPosn ) )
        baseVec = np.multiply( bgnPosn, -1.0 )
        tClose  = np.dot( trvlDir, baseVec )
        closPsn = bgnPosn + np.multiply( trvlDir, tClose )
        dClose  = np.linalg.norm( closPsn )
        rtnPath = [bgnPose,]
        if dClose < _EXTRA_PAD_BASE:
            midPosn = (bgnPosn + endPosn)/2.0
            midDir  = vec_unit( midPosn )
            safPosn = midDir * _EXTRA_PAD_BASE
            safPose = np.eye(4)
            safPose[0:3,0:3] = endPose[0:3,0:3]
            safPose[0:3,3]   = safPosn
            if not self.p_nonneg_Z( safPose ):
                safPose[2,3] = _EXTRA_PAD_BASE
            soln = self.IK( safPose )
            while soln is None:
                safPose[0:3,3] += sample_on_sphere( radius = 0.050 )
                if not self.p_safe_pose( safPose ):
                    soln = None
                else:
                    soln = self.IK( safPose )
            rtnPath.append( safPose )
        rtnPath.append( endPose )
        return rtnPath
    

    def get_pose_energy_func( self, shots, centroid, desiredAngularSeparation_rad : float = 30.0/180.0*np.pi ):

        _CONFIG_FACTOR = 10.0
        _TABLE_FACTOR  = 10.0
        _REACH_FACTOR  =  7.0
        _DELTA_FACTOR  =  2.0
        _DELTA_MAX     = [np.pi for _ in range(6)]
        _DELTA_MAX[-1] = np.pi*2.0
        _DELTA_DIVISOR = np.linalg.norm( _DELTA_MAX )

        def sep_energy( qRef : list | np.ndarray, q : list | np.ndarray ):
            """ Compute badness based on angle between this and existing shots """
            nonlocal shots, centroid, desiredAngularSeparation_rad, self
            pose = LUMP.FK( q )
            vc_i = np.subtract( extract_position( pose ), centroid )
            # print( shots )
            vecs = [np.subtract( extract_position(shot[1]), centroid ) for shot in shots if (shot is not None)]
            vecs.append( np.array([0.0, 0.0, 1.0,]) ) # Penalize being exactly vertical
            angl = [angle_between_vectors_rad(vc_i, vc_f) for vc_f in vecs if (diff_mag( vc_i, vc_f ) > 0.0)]
            nrg  = LUMP.config_energy( qRef, q ) * _CONFIG_FACTOR
            zQ   = pose[2,3]
            nrg += max( 0.0, _RBT_TABLE_MARGIN/max(0.005, zQ) )*_TABLE_FACTOR # Penalize being near the table
            hit = 1.0 if self.p_collision_q( q ) else 0.0
            nrg += hit*_COLLISION_NRG_PENALTY
            nrg += np.linalg.norm( pose[0:2,3] )*_REACH_FACTOR
            nrg += (np.linalg.norm( np.subtract( qRef, q ) )/_DELTA_DIVISOR + abs(qRef[-1] - q[-1])/np.pi)*_DELTA_FACTOR
            nrg += max( 0.0, 0.75 - np.linalg.norm( extract_position( pose ) ) )/0.75*4.0
            for theta_j in angl:
                nrg += max( 0.0, desiredAngularSeparation_rad - theta_j )
            return nrg
        
        return sep_energy
    

    def plan_3d_shots( self, objects : list[GraspObj], dBackup : float, N : int = None, 
                             desiredAngularSeparation_rad : float = 30.0/180.0*np.pi,
                             individual : bool = False ):
        """ A Series of shots with some angular distance between them """
        if N is None:
            N = self.nextNshot
        else:
            self.nextNshot = N
        centroid = None
        if len( objects ):
            centroid = self.symbol_centroid( objects )
        else:
            return None
        shots = []

        if individual:
            for obj_i in objects:
                nuShot = self.plan_3d_shot_centroid( [obj_i,], dBackup, energyFunc = self.get_pose_energy_func( shots, centroid, desiredAngularSeparation_rad ) )
                while nuShot is None:
                    nuShot = self.plan_3d_shot_centroid( [obj_i,], dBackup, energyFunc = self.get_pose_energy_func( shots, centroid, desiredAngularSeparation_rad ) )
                shots.append( nuShot )
        else:
            while len( shots ) < N:
                nuShot = self.plan_3d_shot_centroid( objects, dBackup, energyFunc = self.get_pose_energy_func( shots, centroid, desiredAngularSeparation_rad ) )
                if nuShot is not None:
                    shots.append( nuShot )

        return [np.array( shot[1] ) for shot in shots]
    

    def locate( self, obj : GraspObj ):
        """ Home in on a partcular block """
        if self.robot is None:
            return None
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
        # FIXME: DID THIS EVER WORK? WERE THE POSES CHANGED IN-PLACE?
        if self.robot is None:
            return None
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


    class SearchTarget:
        """ Container class for things we are looking for """
        def __init__( self, thing : GraspObj = None ):
            """ Read from symbol """
            self.pose  = np.eye(4)
            self.label = env_var("_NULL_NAME")
            self.dist  = dict()
            self.count = 0
            if thing is not None:
                self.pose  = extract_pose_as_homog( thing )
                self.label = thing.label
                self.dist  = thing.labels

        def copy( self ):
            """ Return a copy of the object """
            rtnObj = LUMP.SearchTarget()
            rtnObj.pose  = np.array( self.pose )
            rtnObj.label = self.label
            rtnObj.dist  = deepcopy( self.dist )
            rtnObj.count = 0
            return rtnObj

        @staticmethod
        def get_centroid( targets : list[LUMP.SearchTarget] ):
            """ Get the centroid of a collection of targets """
            return np.mean( [extract_position( item.pose ) for item in targets] )

        @staticmethod
        def from_GraspObj_list( things : list[GraspObj] ) -> list[LUMP.SearchTarget]:
            """ Get a list of targets """
            rtnLst = deque()
            for thing in things:
                rtnLst.append( LUMP.SearchTarget( thing ) )
            return list( rtnLst )
        
        @staticmethod
        def propose_gridded_targets( things : list[LUMP.SearchTarget], N : int = 3, gridUnit_m = 0.100 ) -> list[LUMP.SearchTarget]:
            """ Imagine things to look for """
            addLst  = list()
            compass = [[gridUnit_m, 0.0, 0.0,], [-gridUnit_m, 0.0, 0.0,], [0.0, gridUnit_m, 0.0,], [0.0, -gridUnit_m, 0.0,],
                       [0.0, 0.0, -env_var("_BLOCK_SCALE"),], [0.0, 0.0, env_var("_BLOCK_SCALE"),],]
            prob    = 0.25
            while len( addLst ) < N:
                for thing in things:
                    if random() < prob:
                        nuObj = thing.copy()
                        drctn = choice( compass )
                        nuObj.pose[0:3,3] += drctn
                        if random() < 0.5:
                            drctn = choice( compass )
                            nuObj.pose[0:3,3] += drctn
                        if nuObj.pose[2,3] >= 0.0:
                            addLst.append( nuObj )
            return addLst
        
        @staticmethod
        def keep_consistent( things : list[LUMP.SearchTarget], truTargets : list[LUMP.SearchTarget] ):
            rtnLst = list()
            for thing in things:
                thPosn = extract_position( thing.pose )
                found  = False
                for target in truTargets:
                    taPosn = extract_position( target.pose )
                    if diff_mag( thPosn, taPosn ) <= env_var("_BLOCK_SCALE"):
                        rtnLst.append( thing )
                        break
                if (not found) and (random() < 0.5): # 0.25
                    rtnLst.append( thing )
            return rtnLst



        @staticmethod
        def reconcile( things : list[LUMP.SearchTarget], countOverlap = True, decay = 0.70 ):
            """ Merge overlapping targets """
            rtnLst = list()
            banSet = set([])
            Nobj   = len( things )
            for i in range( Nobj-1 ):
                obj_i = things[i]
                psn_i = extract_position( obj_i.pose )
                for j in range( i+1, Nobj ):
                    if j not in banSet:
                        obj_j = things[j]
                        psn_j = extract_position( obj_j.pose )
                        if diff_mag( psn_i, psn_j ) < env_var("_BLOCK_SCALE"):
                            banSet.add(j)
                            if countOverlap:
                                obj_i.count += obj_j.count
            for i in range( Nobj ):
                if (i not in banSet) and (random() < decay):
                    rtnLst.append( things[i] )
            return rtnLst


    @staticmethod
    def sample_covering_shots( targets : list[GraspObj] | list[LUMP.SearchTarget], camDist = dShot, Nshots = 10 ) -> list[np.ndarray]:
        """ Generate a list of shots that will cover as many objects as possible """
        # 1. Get minimum distance that would still fit in the camera frustum
        fovHlf = env_var("_D405_FOV_H_DEG")/180.0 * np.pi / 2.0
        posn   = [extract_position( trgt.pose ) for trgt in targets]
        mean   = np.mean( posn, axis = 0 )
        aabb   = get_aabb( posn )
        sHlf   = np.linalg.norm( np.subtract( aabb[1][:-1], aabb[0][:-1] ) )/2.0
        dMin   = sHlf / np.tan( fovHlf ) * 1.25
        dCam   = max( [env_var("_MIN_CAM_PCD_DIST_M"), dMin, camDist,] ) 
        cPts   = sample_on_sphere( center = mean, radius = dCam, N = Nshots )
        cPos   = sphere_samples_to_eff_poses( cPts, center = mean, effXbasis = [0.0, -1.0, 0.0] )
        rPos   = vary_eff_wrist_3( cPos, N = 3, lo = -np.pi/2.0, hi = np.pi/2.0 )
        return rPos


    @staticmethod
    def wrap_shots( pose, score = None ) -> list[dict] | dict:
        """ Put the effector `pose`(s) in a `dict` container """
        if isinstance( pose, list ) and isinstance( pose[0], (np.ndarray, list, deque) ):
            return [LUMP.wrap_shots( item ) for item in pose]
        else:
            return {
                'pose' : np.array( pose ),
                'score': score if (score is not None) else 0.0,
            }


    def plan_object_shots( self, proposedObjects : list[GraspObj], shotDist = dShot, N = None ) -> list[np.ndarray]:
        """ Get ready for object search """
        if N is None:
            N = env_var("_N_PERC_SHOTS")
        _VIEW_PENALTY  = 1.0
        _EDGE_PENALTY  = 0.75
        _MULT_FACTOR   = 7 #10
        _NEAR_SHOT_PEN = 1.25
        
        targets  = list( proposedObjects )
        centroid = np.mean( [extract_position( obj ) for obj in targets], axis = 0 )

        shots = list()
        Nt    = len( targets )
        Nsee  = 2
        Nij   = max( int(N*_MULT_FACTOR / Nsee), 1 )
        for i in range( Nt-1 ):
            for j in range( i+1, Nt ):
                shots.extend( LUMP.sample_covering_shots( [targets[i], targets[j],], shotDist, Nshots = Nij ) )

        ranking = deque()
        nrgFunc = self.get_pose_energy_func( list(), centroid )
        for shot_i in shots:
            shot = LUMP.wrap_shots( shot_i )
            soln = self.IK( shot_i, suppressCache = True )
            if soln is not None:
                score = nrgFunc( self.q, soln )
                for trgt in targets:
                    if not self.p_target_in_cam_view( shot_i, trgt ):
                        score += _VIEW_PENALTY
                    else:
                        score += max([abs( coord ) for coord in self.viewport_coords( shot_i, trgt )]) * _EDGE_PENALTY
                shot['score'] = score
                ranking.append( shot )
        ranking = list( ranking )

        ranking.sort( key = lambda x: x['score'] )
        # Re-Rank Based on Closeness #
        topPose = ranking[0]['pose']
        for shot in ranking[1:]:
            pose_i = shot['pose']
            # print(topPose, pose_i)
            shot['score'] += LUMP._SEP_DIST_M / max( euclidean_distance_between_poses( topPose, pose_i ), 0.005 ) * _NEAR_SHOT_PEN
        ranking.sort( key = lambda x: x['score'] )

        return [item['pose'] for item in ranking[:N]]
    

    def init_object_search( self, senseCB : Callable, rMoveCB : Callable, fetchCB : Callable, checkCB : Callable, noVizCB : Callable ):
        """ Get ready to search """
        # Data #
        self.shots   = list()
        self.ranking = list()
        # Callbacks #
        self.see_cb  = senseCB
        self.mov_cb  = rMoveCB
        self.get_cb  = fetchCB
        self.chk_cb  = checkCB
        self.viz_cb  = noVizCB
        # Constants #
        self._VIZ_THRESH_DOWN = 0.99 #0.98 # 0.85 # 0.95
        

    def p_target_in_cam_view( self, effXform : np.ndarray, target : LUMP.SearchTarget ):
        """ Can the symbol be seen from the given perspective? """
        camXform = np.array( effXform ).dot( self.robot.camXform )
        bounds   = get_D405_FOV_frustum( camXform )
        while hasattr( target, 'pose' ):
            target = target.pose
        qPosn    = target[0:3,3]
        blcRad   = np.sqrt( 3.0 * (env_var("_BLOCK_SCALE")/2.0)**2 )
        return p_sphere_inside_plane_list( qPosn, blcRad, bounds )
    

    def viewport_coords( self, effXform : np.ndarray, target : LUMP.SearchTarget ):
        """ Can the symbol be seen from the given perspective? """
        camXform = np.array( effXform ).dot( self.robot.camXform )
        while hasattr( target, 'pose' ):
            target = target.pose
        diff = posn_from_xform( target ) - posn_from_xform( camXform )
        xBasis, yBasis, zBasis = bases_from_xform( camXform )
        xCoord, yCoord, zCoord = [diff.dot( basis ) for basis in [xBasis, yBasis, zBasis]]
        X = np.rad2deg( np.arctan2( xCoord, zCoord ) ) / env_var("_D405_FOV_H_DEG")
        Y = np.rad2deg( np.arctan2( yCoord, zCoord ) ) / env_var("_D405_FOV_V_DEG")
        return X, Y

    
    def obscurity_list( self, effXform : np.ndarray, targets : list[LUMP.SearchTarget] ):
        """ Get a list of the degree to which each target is obscured """
        wHalf    = env_var("_BLOCK_SCALE") / 2.0
        Ntrgt    = len( targets )
        obscure  = [0.0 for _ in range( Ntrgt )]
        camXform = np.array( effXform ).dot( self.robot.camXform )
        for i in range( Ntrgt-1 ):
            trgt_i = targets[i]
            pose_i = get_pose_attr( trgt_i )
            dist_i = euclidean_distance_between_poses( camXform, pose_i )
            tDir_i = diff_unit_from_pose1_to_pose2( camXform, pose_i )
            dAng_i = np.arctan( wHalf / dist_i )
            for j in range( i+1, Ntrgt ):
                trgt_j = targets[j]
                pose_j = get_pose_attr( trgt_j )
                dist_j = euclidean_distance_between_poses( camXform, pose_j )
                tDir_j = diff_unit_from_pose1_to_pose2( camXform, pose_j )
                dAng_j = np.arctan( wHalf / dist_j )

                ang_ij = angle_between_vectors_rad( tDir_i, tDir_j )
                if (dAng_i + dAng_j) > ang_ij:
                    angOver = dAng_i + dAng_j - ang_ij
                    if dist_i > dist_j:
                        if dAng_i >= ang_ij:
                            angOver_i  = min( ang_ij, dAng_j )
                            angOver_i += max( 0.0, dAng_i-ang_ij )
                        else:
                            angOver_i = angOver
                        obscure[i] += angOver_i / (2.0*dAng_i)
                    else:
                        if dAng_j >= ang_ij:
                            angOver_j  = min( ang_ij, dAng_i )
                            angOver_j += max( 0.0, dAng_j-ang_ij )
                        else:
                            angOver_j = angOver
                        obscure[j] += angOver_j / (2.0*dAng_j)

        for i in range( Ntrgt ):
            obscure[i] = min( obscure[i], 1.0 )

        return obscure


    def rank_search_shots( self ):
        """ Obtain a ranking of all planned shots """
        _EXCLUDE_PENALTY = 0.75
        _REPEAT_PENALTY  = 6.00 # 4.0 # 2.00 # 1.00 # 0.65
        _EDGE_PENALTY    = 0.75
        _NEAR_SHOT_PEN   = 1.50

        self.set_state_from_robot()
        centroid = LUMP.SearchTarget.get_centroid( self.targets )
        nrgFunc  = self.get_pose_energy_func( list(), centroid )

        nuLst = deque()
        for shot in self.shots:
            obsc = self.obscurity_list( shot, self.targets )
            soln = self.IK( shot, suppressCache = True )
            if soln is not None:
                scor = nrgFunc( self.q, soln )
                for i, trgt in enumerate( self.targets ):
                    if self.p_target_in_cam_view( shot, trgt ):
                        scor += trgt.count * (1.0 - obsc[i]) * _REPEAT_PENALTY
                        scor += max([abs( coord ) for coord in self.viewport_coords( shot, trgt )]) * _EDGE_PENALTY
                        scor += obsc[i] * _EXCLUDE_PENALTY
                    else:
                        scor += _EXCLUDE_PENALTY
                nuLst.append( LUMP.wrap_shots( pose = shot, score = scor ) )

        self.ranking = list( nuLst )
        self.ranking.sort( key = lambda x: x['score'] )

        topPose = self.ranking[0]['pose']
        for shot in self.ranking[1:]:
            pose_i = shot['pose']
            shot['score'] += LUMP._SEP_DIST_M / max( euclidean_distance_between_poses( topPose, pose_i ), 0.005 ) * _NEAR_SHOT_PEN

        for shot in self.ranking:
            pose_i = shot['pose']
            for pose_j in self.shotHist:
                shot['score'] += LUMP._SEP_DIST_M / max( euclidean_distance_between_poses( pose_j, pose_i ), 0.005 ) * _NEAR_SHOT_PEN

        self.ranking.sort( key = lambda x: x['score'] )

        self.shots = [item['pose'] for item in self.ranking]


    def effector_pose_from_camera_pose( self, camPose : np.ndarray ):
        """ Get the effector pose from the `camPose` """
        invCamPose = np.linalg.inv( self.robot.camXform )
        # return invCamPose.dot( camPose )
        return camPose.dot( invCamPose )


    def effector_lookAt_pose( self, cenPosn : np.ndarray, camPosn : np.ndarray, camXbasis = [1.0, 0.0, 0.0,] ):
        """ Construct a camera pose that looks at `cenPosn` """
        lookDir = vec_unit( np.subtract( cenPosn, camPosn ) )
        Rmatrix = bases_from_xB_zB( camXbasis, lookDir, asRotMtx = True )
        camPose = homog_xform( Rmatrix, camPosn )
        return self.effector_pose_from_camera_pose( camPose )
    

    def promising_shots_from_history( self, histDQue : deque, dNudge_m : float = 0.070 ):
        """ Look for shots that segmented the blocks, but did not result in PCDs """
        Ndqu     = len( histDQue )
        srch     = True
        idx      = -1
        oldShots = deque()

        while srch:
            datum = None
            while idx > -Ndqu:
                datum = histDQue[ idx ]
                idx  -= 1
                if datum['msg'] == "ObsMeta":
                    break
            for hit_i in datum['data']['hits']:
                pass

                


        


    def run_birds_eye_search( self, centerXY : np.ndarray, zLo : float, zHi : float, Nshots : int = 3 ):
        """ Look at random spots from random points, and return True if we found all the things! """
        # _GRID_HALF_PTS = 1
        hiCntr = np.zeros( (3,) )
        loCntr = np.zeros( (3,) )
        hiCntr[:2] = centerXY
        hiCntr[2]  = zHi
        loCntr[:2] = centerXY
        loCntr[2]  = zLo
        hiPnts = grid_points_on_plane( hiCntr, [0.0,0.0,1.0,], 0.050, [1.0, 0.0, 0.0,], env_var("_N_GRID_HALF_PTS") )
        loPnts = grid_points_on_plane( loCntr, [0.0,0.0,1.0,], 0.050, [1.0, 0.0, 0.0,], env_var("_N_GRID_HALF_PTS") )
        Npoint = hiPnts.shape[0]
        # print( hiPnts.shape )
        eyePts = [ np.array( hiPnts[ choice( list( range( Npoint ) ) ) ] ) for _ in range( Nshots ) ]
        lukPts = [ np.array( loPnts[ choice( list( range( Npoint ) ) ) ] ) for _ in range( Nshots ) ]
        efPose = deque()
        for i in range( Nshots ):
            efPose.append(  self.effector_lookAt_pose( lukPts[i], eyePts[i], camXbasis = [0.0, -1.0, 0.0,] )  )

        if env_var("_USE_GRAPHICS"):
            render_memory_list( 
                objs      = self.get_cb(),
                robotPose = list( efPose )
            )

        for i, robotPose in enumerate( efPose ):
            self.mov_cb( robotPose )
            self.see_cb( i , Nshots )

        return self.chk_cb()


    def run_object_search( self, proposedObjects : list[GraspObj] ):
        """ Be a little more persistent until the objects are found """
        _MULT_FACTOR  =  7 # 10 # 15
        _N_SHOT_ADD   =  5
        _N_SHOT_TOTAL = _N_SHOT_ADD*_MULT_FACTOR 
        _N_INSPECT    =  3
        _N_LOOK       =  env_var("_N_SEARCH_SHOTS")
        _FINGER_LEN_M = 0.100

        ## Run init scan and return early if we found the objects ##
        if self.run_birds_eye_search( 
            centerXY = [ env_var("_MIN_X_OFFSET") + env_var("_X_WRK_SPAN")/2.0, 
                         env_var("_MIN_Y_OFFSET") + env_var("_Y_WRK_SPAN")/2.0, ], 
            # centerXY = [ -0.340-0.100, -0.130-0.100, ], 
            zLo      = 0.0, 
            zHi      = env_var("_Z_SAFE"), 
            # Nshots   = env_var("_N_SEARCH_SHOTS")+1
            Nshots   = env_var("_N_SEARCH_SHOTS")
        ):
            return True

        self.searchActive = True
        self.shotHist     = deque()

        prevSymbols = LUMP.SearchTarget.from_GraspObj_list( proposedObjects )
        
        self.targets = prevSymbols[:]
        nuTgt = LUMP.SearchTarget.propose_gridded_targets( self.targets, env_var("_N_GRID_EXPAND") )
        self.targets.extend( nuTgt )
        
        self.shots = list()
        bgn = 0
        end = _N_INSPECT
        N   = len( self.targets )
        while bgn < N:
            nuShots = LUMP.sample_covering_shots( self.targets[bgn:end], LUMP.dShot, Nshots = _N_SHOT_ADD )
            self.shots.extend( nuShots )
            bgn = end
            end = min( end+_N_INSPECT, N )

        self.rank_search_shots()

        while not self.chk_cb():

            # Lower threshold for finding things
            self.viz_cb( self._VIZ_THRESH_DOWN )

            # 1. Goto top cam shot
            shots = self.shots[:_N_LOOK] 

            if env_var("_USE_GRAPHICS"):
                render_memory_list( 
                    objs    = self.get_cb(),
                    removed = self.targets, 
                    robotPose = shots 
                )

            for i, shot in enumerate( shots ):
                self.mov_cb( shot )
                self.shots = self.shots[1:] # pop front
                self.see_cb( i , len( shots ) )
                for trgt in self.targets:
                    if self.p_target_in_cam_view( shot, trgt ):
                        trgt.count += 1
            # 2. Gen more targets
            beliefs = LUMP.SearchTarget.from_GraspObj_list( self.get_cb() )
            for bel in beliefs:
                bel.count = 1
            self.targets = LUMP.SearchTarget.keep_consistent( self.targets, beliefs )
            self.targets.extend( beliefs )
            nuTgt = LUMP.SearchTarget.propose_gridded_targets( self.targets, 4 )
            self.targets.extend( nuTgt )
            self.targets.extend( prevSymbols )
            self.targets = LUMP.SearchTarget.reconcile( self.targets, countOverlap = False )
            # 3. Gen more shots
            bgn = 0
            end = _N_INSPECT
            N   = len( self.targets )
            while bgn < N:
                nuShots = LUMP.sample_covering_shots( self.targets[bgn:end], LUMP.dShot, Nshots = _N_SHOT_ADD )
                self.shots.extend( nuShots )
                bgn = end
                end = min( end+_N_INSPECT, N )
            # 4. Filter shots
            self.rank_search_shots()
            if len( self.shots ) > _N_SHOT_TOTAL:
                self.shots = self.shots[:_N_SHOT_TOTAL]

            

        self.searchActive = False
        


########## RENDER ROBOT ############################################################################


def plot_DH_robot( dhParamsMatx, qConfig, axesScale = 0.050 ):
    """ Plot the kinematic chain represented by `dhParamsMatx` in `qConfig`, using Open3d """
    # 0. Set up drawing accounting
    geo     = []
    index   = 0
    lastPnt = posn_from_xform( np.eye(4) )
    addSeg  = [ lastPnt.copy().flatten(), ]
    addIdx  = []
    # 1. Generate link frames
    chain = FK_DH_chain( dhParamsMatx, qConfig ) #, baseLink = baseLink, baseQ = baseQ )
    # 2. Fetch base link bases
    [alpha, a, d] = [0.0 for _ in range(3)]
    theta         = 0.0
    [xB, yB, zB]  = bases_from_xform( dh_link_homog( theta, alpha, a, d ) )    
    
    # 3. For each link: Create geometries for frame, a-segment, and d-segment
    for i, frm in enumerate( chain ):
        
        # 4. Create frame geo
        f_i = scene.visuals.XYZAxis()
        # VISPY IS COLUMN-MAJOR
        rot = np.eye(4)
        rot[0:3,0:3] = frm[0:3,0:3]
        vizXfrm = transforms.linear.MatrixTransform( matrix = rot.transpose() )
        vizXfrm.scale( [axesScale,axesScale,axesScale,] )
        vizXfrm.translate( frm[0:3,3] )
        f_i.transform = vizXfrm
        geo.append( f_i )
        
        if i > 0: 
            # 5. Fetch link measurements
            [alpha, a, d] = dhParamsMatx[i-1]
            theta         = qConfig[i-1]
            
            # 6. Paint 'd', if present
            if abs(d) > 0.0:
                nextPnt = np.add( lastPnt, np.multiply(zB, d) )
                addSeg.append( nextPnt.copy().flatten() )
                addIdx.append( [index, index+1] )
                index += 1
                lastPnt = nextPnt.copy()
        
            # 7. Paint 'a', if present
            if abs(a) > 0.0:
                nextPnt = np.add(
                    lastPnt,
                    np.multiply(
                        R_krot( xB, alpha ).dot( R_krot( zB, theta ) ).dot( xB ), 
                        a
                    )
                )
                addSeg.append( nextPnt.copy().flatten() )
                addIdx.append( [index, index+1] )
                index += 1
                lastPnt = nextPnt.copy()
            
            # 8. Fetch frame bases
            [xB, yB, zB]  = bases_from_xform( frm )
    
    # 9. Create link geo
    geo.append( scene.visuals.Line(
        pos     = np.array(addSeg),
        connect = np.array(addIdx),
        color   = [0.0,0.0,0.0,1.0],
    ) )
    
    return geo

########## MAIN ####################################################################################
if __name__ == "__main__":
    mp = LUMP()
    # mp.verify_IK()

    os.system( 'kill %d' % os.getpid() ) 