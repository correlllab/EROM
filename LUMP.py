import cmath, math, os
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

from random import random
from collections import deque
from collections.abc import Callable
from pprint import pprint

import numpy as np
from numpy import linalg

from magpie_control.poses import vec_unit, is_pose_mtrx
from magpie_control.ur5 import UR5_Interface
from aspire.symbols import euclidean_distance_between_symbols, GraspObj, extract_pose_as_homog, extract_position
from aspire.env_config import env_var
from aspire.utils import diff_norm

_RBT_BASE_BUFFER  = 0.200
_RBT_BASE_FACTOR  = 1.250
_RBT_TABLE_MARGIN = 0.070
_REVERSE_QUERIES  = {
    "bluBlock": {'query': "a photo of a small block", 'abbrv': "blu", },
    "ylwBlock": {'query': "a photo of a small block", 'abbrv': "ylw", },
    "grnBlock": {'query': "a photo of a small block", 'abbrv': "grn", },
    "redBlock": {'query': "a photo of a small block", 'abbrv': "red", },
}

##### Globals #####

global mat
mat=np.matrix


########## DH PARAMETERS ###########################################################################


global d1, a2, a3, d4, d5, d6
d1 =  0.1625
a2 = -0.425
a3 = -0.3922
d4 =  0.1333
d5 =  0.0997
d6 =  0.0996

global d, a, alph

d    = mat([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
a    = mat([0 ,-0.425 ,-0.3922 ,0 ,0 ,0])
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

    print( f"Target:\n{desired_pos}" )

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
           return pnt
        else:
            return np.array([1.0, 0.0, 0.0,])
        
    if N == 1:
        return gen_pnt()
    elif N > 1:
        rtnLst = deque()
        for _ in range(N):
            rtnLst.append( gen_pnt() )
        return np.array( list( rtnLst ) )


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


########## MOTION PLANNER ##########################################################################

class LUMP:
    """ [L]imited [U]R5 [M]otion [P]lanner """
    def __init__( self, qInit = None, robot : UR5_Interface = None ):
        """ Set params """
        ## Intenal Scoring ##
        self.q    : np.ndarray = np.array( [0.0 for _ in range(6)] )
        self.pose : np.ndarray = self.FK( self.q )
        ## Problem-Specific ##
        self.ZTableCam = -0.081666 - 0.017
        self.dShot     = 3.00*env_var( "_MIN_CAM_PCD_DIST_M" )
        self.dLoc      = 1.25*env_var( "_MIN_CAM_PCD_DIST_M" )
        self.robot : UR5_Interface = robot
        if isinstance( qInit, (list, np.ndarray) ):
            self.q = np.array( qInit )


    def p_base_safe( self, effPose : np.ndarray ):
        """ Return true if the effector pose is sufficiently far from the base """
        return (euclidean_distance_between_symbols( np.eye(4), effPose ) >= _RBT_BASE_BUFFER)


    def p_nonneg_Z( self, effPose : np.ndarray, margin = _RBT_TABLE_MARGIN ):
        """ Return True if the Z-position is non-negative """
        return ((effPose[2,3] - margin) >= 0.0)
    

    def p_safe_pose( self, effPose : np.ndarray ):
        """ Should the robot even consider this pose? """
        return (self.p_base_safe( effPose ) and self.p_nonneg_Z( effPose ))
    

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
                print( f"UNSAFE: {arr}" )
        eMin = 1e9
        qMin = None
        for soln in qFltr:
            nrg = self.config_energy( self.q, soln )
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
    def FK( q : list | np.ndarray ):
        """ Perform inverse kinematics (deterministic) """
        th = np.matrix( [[q[0]], [q[1]], [q[2]], [q[3]], [q[4]], [q[5]]] )
        c  = [0]
        return HTrans( th, c )
    

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
    

    def plan_3d_shot_centroid( self, objects : list[GraspObj], dBackup : float, N : int = 8, energyFunc : Callable = None ):
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
            print( rtnSoln )
            if ((rtnSoln is not None) and self.p_safe_pose( rtnPose )):
                ranking.append((
                    energyFunc( self.q, rtnSoln ),
                    np.array( rtnSoln ),
                ))
            else:
                print( f"Cannot Rank: {rtnSoln}" )

        ranking = list( ranking )
        ranking.sort( key = lambda x: x[0], reverse = True )

        if len( ranking ):
            return ranking[0][1], self.FK( ranking[0][1] )
        else:
            return None
    

    def make_path_safe( self, bgnPose, endPose ):
        """ If the straight-line path would pass too close to the robot base, then propose an intermediate waypoint """
        bgnPosn = extract_position( bgnPose )
        endPosn = extract_position( endPose )
        trvlDir = vec_unit( np.subtract( endPosn, bgnPosn ) )
        baseVec = np.multiply( bgnPosn, -1.0 )
        tClose  = np.dot( trvlDir, baseVec )
        closPsn = bgnPosn + np.multiply( trvlDir, tClose )
        dClose  = np.linalg.norm( closPsn )
        rtnPath = [bgnPose,]
        if dClose < _RBT_BASE_BUFFER:
            midPosn = (bgnPosn + endPosn)/2.0
            midDir  = vec_unit( midPosn )
            safPosn = midDir * (_RBT_BASE_BUFFER * _RBT_BASE_FACTOR)
            safPose = np.eye(4)
            safPose[0:3,0:3] = endPose[0:3,0:3]
            safPose[0:3,3]   = safPosn
            if not self.p_nonneg_Z( safPose ):
                safPose[2,3] = _RBT_TABLE_MARGIN * _RBT_BASE_FACTOR
            rtnPath.append( safPose )
        rtnPath.append( endPose )
        return rtnPath
    

    def plan_3d_shots( self, objects : list[GraspObj], dBackup : float, N : int, desiredAngularSeparation_rad : float = 30.0/180.0*np.pi ):
        """ A Series of shots with some angular distance between them """
        centroid = None
        if len( objects ):
            centroid = self.symbol_centroid( objects )
        else:
            return None
        shots = []

        def sep_energy( qRef : list | np.ndarray, q : list | np.ndarray ):
            """ Compute badness based on angle between this and existing shots """
            nonlocal shots, centroid, desiredAngularSeparation_rad
            pose = LUMP.FK( q )
            vc_i = np.subtract( extract_position( pose ), centroid )
            print( shots )
            vecs = [np.subtract( extract_position(shot[1]), centroid ) for shot in shots if (shot is not None)]
            angl = [angle_between_vectors_rad(vc_i, vc_f) for vc_f in vecs]
            nrg  = LUMP.config_energy( qRef, q )
            for theta_j in angl:
                nrg += max( 0.0, desiredAngularSeparation_rad - theta_j )
            return nrg

        while len( shots ) < N:
            nuShot = self.plan_3d_shot_centroid( objects, dBackup, energyFunc = sep_energy )
            if nuShot is not None:
                shots.append( nuShot )

        return shots
    

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


    # def verify_IK( self ):
    #     """ Is our IK any good? """
    #     jnts = np.array([ -1.0+2.0*random() for _ in range(6) ])
    #     pose = self.FK( jnts )
    #     print( f"Solve for effector pose:\n{pose}" )
    #     soln = self.IK( pose )
    #     if soln is not None:
    #         sPos = self.FK( soln )
    #         print( f"Sol'n : {soln}" )
    #         print( f"Config: {jnts}" )
    #         diff = euclidean_distance_between_symbols( pose, sPos )
    #         print( f"Difference between actual and IK sol'n: {diff}" )
    #     else:
    #         print( "FAILED to solve!" )
        


########## MAIN ####################################################################################
if __name__ == "__main__":
    mp = LUMP()
    # mp.verify_IK()

    os.system( 'kill %d' % os.getpid() ) 