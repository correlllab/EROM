import cmath, math
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

from copy import copy

import numpy as np
from numpy import linalg

from magpie_control.ur5 import UR5_Interface
from aspire.symbols import euclidean_distance_between_symbols, extract_pose_as_homog

_RBT_BASE_BUFFER = 0.200

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
#TODO: change these too

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


def isJointPosSafe( joints , plane ):
    """ #a*x+b*y+c*z+d """
    
    # do fwd kinematics
    th     = np.matrix( [[joints[0]], [joints[1]], [joints[2]], [joints[3]], [joints[4]], [joints[5]]] )
    c      = [0]
    fk     = HTrans(th, c)
    coords = [t[0] for t in fk[:3,3].tolist()]
    
    for i in range(1,7):
        fk     = HTrans(th, c, i)
        coords = [t[0] for t in fk[:3,3].tolist()]
        val    = (plane[0] * coords[0]) + (plane[1] * coords[1]) + (plane[2] * coords[2]) + plane[3]
        
        if val >= 0:
            return False    
    return True



########## INVERSE KINEMATICS ######################################################################
    
def invKine( desired_pos ):# T60
    """ #inverse kinematics pasted here
        #KEEP ROWS AND COLUMNS STRAIGHT
        #each column is a result to check
        #8 RESULTS, 6 JOINTS  """
    th   = mat( np.zeros((6, 8)) )
    P_05 = ( desired_pos * mat([0,0, -d6, 1]).T-mat([0,0,0,1 ]).T )

    # **** theta1 ****

    psi = atan2(P_05[2-1,0], P_05[1-1,0])
    phi = acos(d4 /sqrt(P_05[2-1,0]*P_05[2-1,0] + P_05[1-1,0]*P_05[1-1,0]))
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



########## MOTION PLANNER ##########################################################################

class LUMP:
    """ [L]imited [U]R5 [M]otion [P]lanner """
    def __init__( self, robot = None ):
        self.robot : UR5_Interface = robot

    def p_base_safe( self, effPose : np.ndarray ):
        """ Return true if the effector pose is sufficiently far from the base """
        return (euclidean_distance_between_symbols( np.eye(4), effPose ) >= _RBT_BASE_BUFFER)
    
    def p_nonneg_Z( self, effPose : np.ndarray ):
        """ Return True if the Z-position is non-negative """
        return (effPose[2,3] >= 0.0)
    
    def IK( self, effPose : np.ndarray ):
        """ Perform inverse kinematics (deterministic) """
        return invKine( effPose )