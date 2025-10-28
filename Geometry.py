########## INIT ####################################################################################

import numpy as np

from magpie_control.poses import vec_unit, repair_pose
from magpie_control.homog_utils import R_x, R_y, posn_from_xform

from aspire.env_config import env_var
from aspire.symbols import ( GraspObj, extract_pose_as_homog, )


########## 3D GEOMETRY #############################################################################

def closest_ray_points( A_org, A_dir, B_org, B_dir ):
    """ Return (closest on ray A to ray B), (closest on ray B to ray A), and their mean point """
    # https://palitri.com/vault/stuff/maths/Rays%20closest%20point.pdf
    c  = np.subtract( B_org, A_org )
    aa = np.dot( A_dir, A_dir )
    bb = np.dot( B_dir, B_dir )
    ab = np.dot( A_dir, B_dir )
    ac = np.dot( A_dir, c     )
    bc = np.dot( B_dir, c     ) 
    dv = (aa*bb-ab*ab)
    if abs( dv ) < 0.001:
        return None, None, None
    fA = (-ab*bc + ac*bb) / dv
    fB = ( ab*ac - bc*aa) / dv
    pA = np.add( A_org, np.multiply( A_dir, fA ) )
    pB = np.add( B_org, np.multiply( B_dir, fB ) )
    return pA, pB, (pA + pB)/2.0



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


def point_above_plane( qPnt, pPnt, pNrm, margin = 0.000 ):
    """ Return True if `qPnt` is on the `pNrm` side of `pPnt`, including `margin` """
    return (np.dot( np.subtract( qPnt, pPnt ), vec_unit( pNrm ) ) >= margin)


def p_sphere_inside_plane_list( qCen, qRad, planeList ):
    """ Return True if a sphere with `qCen` and `qRad` can be found above every plane in `planeList` = [ ..., [point, normal], ... ] """
    if len( qCen ) == 4:
        qCen = posn_from_xform( qCen )
    for (pnt_i, nrm_i) in planeList:
        if not point_above_plane( qCen, pnt_i, nrm_i, margin = qRad ):
            return False
    return True


def p_symbol_in_cam_view( camXform : np.ndarray, symbol : GraspObj ):
    """ Can the symbol be seen from the given perspective? """
    bounds = get_D405_FOV_frustum( camXform )
    qPosn  = extract_pose_as_homog( symbol )[0:3,3]
    blcRad = np.sqrt( 3.0 * (env_var("_BLOCK_SCALE")/2.0)**2 )
    return p_sphere_inside_plane_list( qPosn, blcRad, bounds )


def bases_from_xB_zB( xBasis : np.ndarray, zBasis : np.ndarray, asRotMtx = False ):
    """ Get basis vectors from a defined `zBasis` and a preferred `xBasis` """
    zBasis = vec_unit( zBasis )
    xBasis = vec_unit( xBasis )
    yBasis = vec_unit( np.cross( zBasis, xBasis ) )
    xBasis = vec_unit( np.cross( yBasis, zBasis ) )
    rBases = xBasis, yBasis, zBasis
    if asRotMtx:
        rtnMtx = np.zeros( (3,3,) )
        for i, basis in enumerate( rBases ):
            rtnMtx[:,i] = basis
        return rtnMtx
    else:
        return rBases


def grid_points_on_plane( center : np.ndarray, normal : np.ndarray, unit_m : float, xBasis : np.ndarray, Nhalf : int ) -> np.ndarray:
    """ Create a regular square grid of 3D points on a plane """
    center = np.array( center )
    xBasis, yBasis, _ = bases_from_xB_zB( xBasis, normal )
    # print( "Bases:", xBasis, yBasis )
    N      = Nhalf*2+1
    rtnArr = np.zeros( (N**2,3,) )
    k      = 0

    def rm_neg_zero( vec : np.ndarray ):
        """ Remove negative zeros """
        rntLst = vec.tolist()
        return np.array( [elem if abs(elem) > 0.0001 else 0.0 for elem in rntLst] )

    for i in range( -Nhalf, Nhalf+1 ):
        Xi = rm_neg_zero( np.multiply( xBasis, i*unit_m ) )
        for j in range( -Nhalf, Nhalf+1 ):
            Yj = rm_neg_zero( np.multiply( yBasis, j*unit_m ) )
            # print( f"\t{center}  +  {Xi}  +  {Yj}  =  {center + Xi + Yj}" )
            rtnArr[k,:] = center + Xi + Yj
            k += 1
    return rtnArr
