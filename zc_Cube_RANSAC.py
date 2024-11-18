import numpy as np

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


def get_normals( vertTriples : np.ndarray ):
    """ Return the normal at each vertex, assuming every three vertices form a triangle """
    rtnNrm = np.ones( vertTriples.shape )
    for i in range( 0, len( vertTriples ), 3 ):
        p0 = vertTriples[i  ,:3]
        p1 = vertTriples[i+1,:3]
        p2 = vertTriples[i+2,:3]
        v0 = np.subtract( p1, p0 )
        v1 = np.subtract( p2, p1 )
        ni = np.cross( v0, v1 )
        rtnNrm[ i:i+3, :3 ] = [ni, ni, ni,]
    return rtnNrm


def min_dist_to_mesh( q, verts ):
    """ Given a list of triangle verts, Return the least distance to the mesh """
    # HACK: This function does NOT take into account the distance to the triangle edge in the case that it is the least distance
    norms = get_normals( verts )
    diffs = np.subtract( verts, q )
    # NOTE: This computes the distance to each plane 3 times
    plnDs = np.sum( norms * diffs, axis = 1, keepdims = True ) # https://stackoverflow.com/q/62500584
    pntDs = np.linalg.norm( diffs, axis = 1 ) # https://stackoverflow.com/a/7741976
    return min( np.min( plnDs ), np.min( pntDs ) )