import numpy as np

from aspire.homog_utils import vec_unit

def get_cube( sideLen, homog ):
    """ Return a 3x36 array of cube vertices, transformed by `homog` """
    hl = sideLen/2.0
    #                Front
    vrts = np.array([[-hl,-hl, hl, 1.0],
                     [+hl,-hl, hl, 1.0],
                     [+hl,+hl, hl, 1.0],
         
                     [+hl,+hl, hl, 1.0],
                     [-hl,+hl, hl, 1.0],
                     [-hl,-hl, hl, 1.0],
         
                     # Back
                     [+hl,-hl,-hl, 1.0],
                     [-hl,-hl,-hl, 1.0],
                     [-hl,+hl,-hl, 1.0],
         
                     [-hl,+hl,-hl, 1.0],
                     [+hl,+hl,-hl, 1.0],
                     [+hl,-hl,-hl, 1.0],
         
                     # Right
                     [+hl,-hl,+hl, 1.0],
                     [+hl,-hl,-hl, 1.0],
                     [+hl,+hl,-hl, 1.0],
         
                     [+hl,+hl,-hl, 1.0],
                     [+hl,+hl,+hl, 1.0],
                     [+hl,-hl,+hl, 1.0],
         
                     # Left
                     [-hl,-hl,-hl, 1.0],
                     [-hl,-hl,+hl, 1.0],
                     [-hl,+hl,+hl, 1.0],
         
                     [-hl,+hl,+hl, 1.0],
                     [-hl,+hl,-hl, 1.0],
                     [-hl,-hl,-hl, 1.0],
         
                     # Top
                     [-hl,+hl,+hl, 1.0],
                     [+hl,+hl,+hl, 1.0],
                     [+hl,+hl,-hl, 1.0],
         
                     [+hl,+hl,-hl, 1.0],
                     [-hl,+hl,-hl, 1.0],
                     [-hl,+hl,+hl, 1.0],
         
                     # Bottom
                     [-hl,-hl,-hl, 1.0],
                     [+hl,-hl,-hl, 1.0],
                     [+hl,-hl,+hl, 1.0],
         
                     [+hl,-hl,+hl, 1.0],
                     [-hl,-hl,+hl, 1.0],
                     [-hl,-hl,-hl, 1.0],])
    return np.dot( homog, vrts.transpose() )[:3,:].transpose()


def get_normals_diameters_and_centers( vertTriples : np.ndarray ):
    """ Return the normal (and "diameter") at each vertex, assuming every three vertices form a triangle """
    # NOTE: Diameters will be used a distance heuristic
    # NOTE: This function outputs some distances 3 times in order to make the Numpy operations nice
    rtnNrm = np.ones( vertTriples.shape )
    rtnCen = np.ones( vertTriples.shape )
    rtnDia = np.ones( len( vertTriples ) )
    for i in range( 0, len( vertTriples ), 3 ):
        p0 = vertTriples[i  ,:3]
        p1 = vertTriples[i+1,:3]
        p2 = vertTriples[i+2,:3]
        v0 = np.subtract( p1, p0 )
        v1 = np.subtract( p2, p1 )
        v2 = np.subtract( p0, p2 )
        mx = np.max( [np.linalg.norm( v0 ), np.linalg.norm( v1 ), np.linalg.norm( v2 ),] )
        ni = vec_unit( np.cross( v0, v1 ) )
        ci = ( p0 + p1 + p2 ) / 3.0
        rtnNrm[ i:i+3, :3 ] = [ni, ni, ni,]
        rtnCen[ i:i+3, :3 ] = [ci, ci, ci,]
        rtnDia[ i:i+3 ] = [mx, mx, mx,]
    return rtnNrm, rtnDia, rtnCen


def min_dist_to_mesh( q, verts, norms, diams, cntrs ):
    """ Given a list of triangle verts, Return the least distance from `q` to the mesh, HACK: We don't actually care if it's very accurate! """
    # HACK: This function uses a distance heuristic to determine if the point distance or plane distance is correct
    # HACK: This function does NOT take into account the distance to the triangle edge in the case that it is the least distance
    # NOTE: This function computes some distances 3 times in order to make the Numpy operations nice
    factr = 1.5 # Bubble factor
    Npnts = len( verts )
    diffs = np.subtract( verts, q )
    cenDf = np.subtract( cntrs, q )
    cenDs = np.linalg.norm( cenDf, axis = 1 )
    plnDs = np.sum( norms * diffs, axis = 1, keepdims = True ) # https://stackoverflow.com/q/62500584
    pntDs = np.linalg.norm( diffs, axis = 1 ) # https://stackoverflow.com/a/7741976
    dMin  = 1e9
    for i in range( Npnts ):
        # HACK: CHOOSE POINT DISTANCE IF OUTSIDE OF BUBBLE, CHOOSE PLANE DISTANCE IF INSIDE BUBBLE
        if (cenDs[i,0] > (diams[i]*factr)):
            dMin = min( dMin, pntDs[i,0] )
        else:
            dMin = min( dMin, plnDs[i,0] )
    return dMin


def approx_cube_RANSAC( points, sideLen, seed = None ):
    """ Fit the pose of a cube of size `sideLen` to `points`, using a slightly inaccurate distance model """
    Npnts = len( points )
    cPose = np.eye(4)
    dPnts = np.zeros( Npnts )
    if seed is not None:
        cPose = seed
    else:
        cPose[0:3,3] = np.mean( points, axis = 0 )

    ##### FIXME: IMPLEMENT RANSAC #####
    
    ## Get model at pose ##
    verts = get_cube( sideLen, cPose )
    norms, diams, cntrs = get_normals_diameters_and_centers( verts )
    
    ## Calc distances to model ##
    for i, q in enumerate( points ):
        dPnts[i] = min_dist_to_mesh( q, verts, norms, diams, cntrs )



if __name__ == "__main__":
    # FIXME, START HERE: TEST A FEW POINTS WITH A UNIT CUBE AT THE ORIGIN (DRAW IT?)
    pass

    # FIXME: TEST RANSAC WITH A NOISY CUBE OF POINTS
    # FIXME: GENERATE A PARTIAL PCD AS FROM A MASKED OBLIQUE VIEW
    # FIXME: TEST RANSAC WITH A MASKED OBLIQUE VIEW