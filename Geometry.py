########## INIT ####################################################################################

import numpy as np

from magpie_control.poses import vec_unit, repair_pose
from magpie_control.homog_utils import R_x, R_y, posn_from_xform

from aspire.env_config import env_var
from aspire.symbols import ( ObjPose, GraspObj, extract_pose_as_homog, )



########## SENSOR PLACEMENT ########################################################################

def get_D405_FOV_frustum( camXform ):
    """ Get 5 <point, normal> pairs for planes bounding an Intel RealSense D405 field of view with its focal point at `camXform` """
    ## Fetch Components ##
    rtnFOV   = list()
    camXform = repair_pose( camXform ) # Make sure all bases are unit vectors
    camRot   = camXform[0:3,0:3]
    cFocus   = camXform[0:3,3]
    ## Depth Limit ##
    dNrm = camRot.dot( [0.0,0.0,-1.0,] )
    dPos = np.eye(4)
    dPos[2,3] = env_var("_D405_FOV_D_M")
    dPnt = camXform.dot( dPos )[0:3,3]
    rtnFOV.append( [dPnt, dNrm,] )
    ## Top Limit ##
    tNrm = camRot.dot( R_x( -np.radians( env_var("_D405_FOV_V_DEG")/2.0 ) ).dot( [0.0,-1.0,0.0] ) )
    tPnt = cFocus.copy()
    rtnFOV.append( [tPnt, tNrm,] )
    ## Bottom Limit ##
    bNrm = camRot.dot( R_x( np.radians( env_var("_D405_FOV_V_DEG")/2.0 ) ).dot( [0.0,1.0,0.0] ) )
    bPnt = cFocus.copy()
    rtnFOV.append( [bPnt, bNrm,] )
    ## Right Limit ##
    rNrm = camRot.dot( R_y( np.radians( env_var("_D405_FOV_H_DEG")/2.0 ) ).dot( [1.0,0.0,0.0] ) )
    rPnt = cFocus.copy()
    rtnFOV.append( [rPnt, rNrm,] )
    ## Left Limit ##
    lNrm = camRot.dot( R_y( -np.radians( env_var("_D405_FOV_H_DEG")/2.0 ) ).dot( [-1.0,0.0,0.0] ) )
    lPnt = cFocus.copy()
    rtnFOV.append( [lPnt, lNrm,] )
    ## Return Limits ##
    return rtnFOV


def p_sphere_inside_plane_list( qCen, qRad, planeList ):
    """ Return True if a sphere with `qCen` and `qRad` can be found above every plane in `planeList` = [ ..., [point, normal], ... ] """
    if len( qCen ) == 4:
        qCen = posn_from_xform( qCen )
    for (pnt_i, nrm_i) in planeList:
        # print(pnt_i, nrm_i)
        dif_i = np.subtract( qCen, pnt_i )
        dst_i = np.dot( dif_i, vec_unit( nrm_i ) )
        # print( f"Distance to Plane: {dst_i}" )
        if dst_i < qRad:
            return False
    return True


def p_symbol_in_cam_view( camXform : np.ndarray, symbol : GraspObj ):
    """ Can the symbol be seen from the given perspective? """
    bounds = get_D405_FOV_frustum( camXform )
    qPosn  = extract_pose_as_homog( symbol )[0:3,3]
    blcRad = np.sqrt( 3.0 * (env_var("_BLOCK_SCALE")/2.0)**2 )
    return p_sphere_inside_plane_list( qPosn, blcRad, bounds )