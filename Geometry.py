########## INIT ####################################################################################

##### Imports #####

### Standard ###
from collections import deque
from random import random

### Special ###
import numpy as np

### Local ###
from magpie_control.poses import vec_unit
from magpie_control.homog_utils import R_krot
from magpie_control.utils import vec_diff_mag



########## GEOMETRY HELPERS ########################################################################
# Neighbor will see vertices of the border segment in reverse order
_tri_indices_RH = [ ( 0 , 1 ) , ( 1 , 2 ) , ( 2 , 0 ) ] # Right-hand indices
_tri_indices_LH = [ ( 1 , 0 ) , ( 2 , 1 ) , ( 0 , 2 ) ] # Left-hand  indices 


def tri_normal( p0, p1, p2 ):
    """ Return the unit normal vector for a triangle with points specified in CCW order """
    vec1 = np.subtract( p1 , p0 )
    vec2 = np.subtract( p2 , p0 )
    return vec_unit( np.cross( vec1 , vec2 ) )


def VF_to_N( verts , facets ):
    """ Given a list of vertices and a list of facets, return N perpendicular to each facet """
    N = deque()
    for f_i in facets:
        p_i = [ verts[j] for j in f_i ]
        N.append( tri_normal( p_i[0] , p_i[1] , p_i[2] ) )
    return np.array( list(N) )


def ray_dir( absOrg, absPnt ):
    """ Return the unit direction `absOrg`--to->`absPnt` """
    return vec_unit( np.subtract( absPnt, absOrg ) )


def facet_adjacency_list_ordered( F ):
    """ Given a facet vertex lookup matrix 'F', Find all of the side-sharing neighbors of each facet, ORDERED VERSION """
    N = len( F ) # Number of facets
    neighborList = [ [ None , None , None ] for i in range( N ) ] # Ordered list of neighbors ( neighbor pointers )
    for i in range( N - 1 ):
        # Copy right-hand edges for each neighbor position that has not yet been associated with a neighbor
        neighbors = [ _tri_indices_RH[ checkDex ] if neighborList[i][ checkDex ] == None else None for checkDex in range(3) ] # Avoid repeat search
        for j in range( i + 1 , N ): # For each unique pairing of facets ( i , j )
            for nDex , neighbor in enumerate( neighbors ):
                if neighbor != None: # If we have not yet located the neighbor for this edge , Check needed to avoid repeats , see above
                    for nnDex , neighborNeighbor in enumerate( _tri_indices_LH ):
                        # nDex : This facet's edge, nnDex : Candidate neighbor edge
                        if neighborList[ j ][ nnDex ] == None:
                            match = True
                            for pairDex in range( 2 ): # For each of the points that form the candidate border
                                # Unless this edge has two vertices that are in reverse order of the other facet edge , this edge is not the 
                                #  border between the two
                                if F[ i ][ neighbor[ pairDex ] ] != F[ j ][ neighborNeighbor[ pairDex ] ]:
                                    match = False
                            if match:
                                neighborList[ i ][ nDex  ] = j # Mark the located neighbor
                                neighbors[ nDex ] = None # Mark this neighbor as found
                                neighborList[ j ][ nnDex ] = i # This facet is the neighbor's neighbor at its identified border
                            # else no match , no action , continue
                        # else the neighbor already has a match for this neighbor-nerighbor
                # else 'neighbor' is Null , we found this neighbor already and there is no action
    return neighborList



########## NUMPY MESH (VFN) ########################################################################

class npVFN:
    """ Simple Mesh made of `np.ndarray`s of [V]ertices, [F]aces, and [N]ormals """

    def build_from_unshared_vertices( self, V ):
        """ Populate mesh assuming an ordered list of vertices """
        pass


    def __init__( self, V = None, F = None, N = None ):
        """ Set arrays """
        self.V = np.array( V, dtype = float ) if (V is not None) else np.zeros( (0,3), dtype = float ) # Vertices
        self.F = np.array( F, dtype = int   ) if (F is not None) else np.zeros( (0,3), dtype = int   ) # Faces
        self.N = np.array( N, dtype = float ) if (N is not None) else np.zeros( (0,3), dtype = float ) # Normals
        self.B = None # -------------------------------------------------------------------------------- Bounding Box
        
    
    def __len__( self ):
        """ Return the number of triangles """
        return len( self.F ) 


    def get_faces_as_tris( self ) -> np.ndarray:
        """ Get all faces as triangles """
        rtnFaces = np.zeros( self.F.shape[0], self.F.shape[1], 3 )
        for i, face in enumerate( self.F ):
            for j, idx in enumerate( face ):
                rtnFaces[i,j,:] = self.V[ idx, : ]
        return rtnFaces


    def add_tri( self, p0, p1 = None, p2 = None ):
        """ Add a triangle to the mesh, **without** shared vertices """
        p0 = np.array( p0 )
        # If we got a triple of points
        if len( p0.shape ) > 1:
            nuTri = p0
            p0, p1, p2 = p0
        # If we got three separate points
        else:
            nuTri = np.array([p0, p1, p2])

        self.V = np.vstack( (self.V, nuTri) )
        index  = len( self ) * 3
        self.F = np.vstack( (self.F, [index, index+1, index+2]) )
        self.N = np.vstack( (self.N, tri_normal( p0, p1, p2 )) )
    

    def get_aabb( self ):
        """ Get the Axis-Aligned Bounding Box """
        aabb = np.array( [[1e9 for _ in range(3)],[-1e9 for _ in range(3)],] )
        for vtx in self.V:
            # Min Corner #
            aabb[0,0] = np.min( aabb[0,0], vtx[0] )
            aabb[0,1] = np.min( aabb[0,1], vtx[1] )
            aabb[0,2] = np.min( aabb[0,2], vtx[2] )
            # Max Corner #
            aabb[1,0] = np.max( aabb[1,0], vtx[0] )
            aabb[1,1] = np.max( aabb[1,1], vtx[1] )
            aabb[1,2] = np.max( aabb[1,2], vtx[2] )
        self.B = aabb.copy()
        return aabb
    

    def get_approx_width( self ):
        """ Get the distance between the corners of the AABB """
        aabb = self.get_aabb()
        return vec_diff_mag( aabb[1,:], aabb[0,:] )

    
    def get_occlusion_frustum( self, viewPoint : np.ndarray, dMax : float, numDiv = 6 ):
        """ Get a (approx. hexagonal prism) region blocked by this object from the given viewpoint """
        rtn = npVFN()
        rad = (self.get_approx_width() / 2.0) * 0.85
        ctr = (self.B[0,:] + self.B[1,:])/2.0
        axs = ray_dir( viewPoint, ctr )
        cbt = ctr + (axs*dMax)
        bgn = np.cross( axs, vec_unit([random() for _ in range(3)]) )*rad + ctr
        bbg = bgn + (ray_dir( viewPoint, bgn )*dMax)
        tLs = bgn
        bLs = bbg
        for i in range( 1, numDiv+1 ):
            t_i = ctr + R_krot( axs, i*(2.0*np.pi/numDiv)).dot( (bgn-ctr) ) 
            b_i = cbt + R_krot( axs, i*(2.0*np.pi/numDiv)).dot( (bbg-cbt) ) 
            rtn.add_tri( ctr, t_i, tLs ) # Top
            rtn.add_tri( tLs, t_i, b_i ) # Side 1/2
            rtn.add_tri( tLs, b_i, bLs ) # Side 2/2
            rtn.add_tri( cbt, bLs, b_i ) # Side 2/2
            tLs = t_i.copy()
            bLs = b_i.copy()
        return rtn
    

    def copy( self ):
        """ Get a copy of this mesh """
        return npVFN( self.V.copy(), self.F.copy(), self.N.copy() )
    

    def copy_as_bounds( self ):
        """ Copy the mesh, Reverse the normals, and Return """
        rtnVFN = self.copy()
        rtnVFN.N = -rtnVFN.N
        return rtnVFN
    

    def p_inside( self, q ):
        """ Is `q` inside of this mesh, NOTE: Assumes *inward*-facing normals! """
        for i, fDices in enumerate( self.F ):
            nrm_i = self.N[i,:]
            pnt_i = self.V[ fDices[0], : ]
            # FIXME, START HERE: TEST EACH FACET!




########## COMMON SOLIDS ###########################################################################

def make_cuboid( xLen, yLen, zLen ):
    """ Return a cube mesh """
    hX  = xLen / 2.0
    hY  = yLen / 2.0
    hZ  = zLen / 2.0
    cbd = npVFN()
    # /// Load Vertices ///
    cbd.V = np.zeros( (8, 3,), dtype = float )
    cbd.V[0,:] = np.array( [-hX, -hY, -hZ,] )
    cbd.V[1,:] = np.array( [ hX, -hY, -hZ,] )
    cbd.V[2,:] = np.array( [ hX,  hY, -hZ,] )
    cbd.V[3,:] = np.array( [-hX,  hY, -hZ,] )
    cbd.V[4,:] = np.array( [-hX, -hY,  hZ,] )
    cbd.V[5,:] = np.array( [ hX, -hY,  hZ,] )
    cbd.V[6,:] = np.array( [ hX,  hY,  hZ,] )
    cbd.V[7,:] = np.array( [-hX,  hY,  hZ,] )
    # /// Load Faces ///
    cbd.F = np.zeros( (8, 3,), dtype = int )
    cbd.F[ 0,:] = np.array( [0, 2, 1,] )
    cbd.F[ 1,:] = np.array( [0, 3, 2,] )
    cbd.F[ 2,:] = np.array( [0, 1, 5,] )
    cbd.F[ 3,:] = np.array( [0, 5, 4,] )
    cbd.F[ 4,:] = np.array( [0, 4, 3,] )
    cbd.F[ 5,:] = np.array( [3, 4, 7,] )
    cbd.F[ 6,:] = np.array( [3, 7, 2,] )
    cbd.F[ 7,:] = np.array( [2, 7, 6,] )
    cbd.F[ 8,:] = np.array( [2, 6, 1,] )
    cbd.F[ 9,:] = np.array( [1, 6, 5,] )
    cbd.F[10,:] = np.array( [6, 7, 5,] )
    cbd.F[11,:] = np.array( [5, 7, 4,] )
    # /// Load Normals ///
    cbd.N = VF_to_N( cbd.V , cbd.N )
