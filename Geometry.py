########## INIT ####################################################################################

##### Imports #####

### Standard ###
from collections import deque

### Special ###
import numpy as np

### Local ###
from magpie_control.poses import vec_unit
from magpie_control.homog_utils import vec_angle_between
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


def tris_to_quad( triA, triB ):
    """ Attempt to merge 2 triangles into a quad """
    quad = list()
    shA = shB = None
    # 1. Find the shared edge
    for idxRH in _tri_indices_RH:
        a0 = triA[ idxRH[0], : ]
        a1 = triA[ idxRH[1], : ]
        for idxLH in _tri_indices_LH:
            b0 = triB[ idxLH[0], : ]
            b1 = triB[ idxLH[1], : ]
            if ((vec_diff_mag( a0, b0 ) + vec_diff_mag( a1, b1 )) < 0.00002):
                shA = list( idxRH )
                shB = [idxLH[1], idxLH[0],]
                break
        if shA is not None:
            break
    if shA is None:
        return None
    # 2. Construct the quad
    iA = shA[1]
    iB = shB[1]
    for _ in range(2):
        quad.append( triA[iA,:] )
        iA = (iA+1)%3
    for _ in range(2):
        quad.append( triB[iB,:] )
        iB = (iB+1)%3
    return np.array( quad )


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
        self.Q = None # -------------------------------------------------------------------------------- Quads
        
    
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


    def get_visible_submesh( self, viewPoint : np.ndarray ):
        """ Get all surfaces with a normal facing the `viewPoint` """
        faces  = self.get_faces_as_tris()
        rtnVFN = npVFN()
        for i, nrm_i in enumerate( self.N ):
            tri_i = faces[i]
            dir_i = np.subtract( viewPoint, tri_i[0] )
            if np.dot( dir_i, nrm_i ) > 0.0:
                rtnVFN.add_tri( tri_i )
        return rtnVFN
    

    def find_quads( self ):
        """ Get neighboring triangles that have a low angle between their normals """
        tris = self.get_faces_as_tris()
        adjc = facet_adjacency_list_ordered( self.F )
        quad = deque()
        for i in range( len( self ) ):
            norm_i = self.N[i,:]
            for j in adjc[i]:
                if j is not None:
                    norm_j = self.N[j,:]
                    if (vec_angle_between( norm_i, norm_j ) < 0.00001):
                        tri_i = tris[i,:,:]
                        tri_j = tris[j,:,:]




    def erase_shared_quads( self ):
        """ Eliminate interior faces """
        # FIXME: ERASE TRIANGLES THAT PARTICIPATE IN SHARED QUADS
        pass


    def get_occlusion_frustum( self, viewPoint : np.ndarray, dMax : float ):
        """ Get a region blocked by this object from the given viewpoint """
        surf  = self.get_visible_submesh( viewPoint )
        faces = surf.get_faces_as_tris()
        pt0 = pt1 = pt2 = pt3 = None
        for tri_i in faces:
            farTri = list()
            for edge_j in range(3):
                # Add quad projected from this edge #
                pt0 = tri_i[ edge_j ]
                pt1 = tri_i[ (edge_j+1)%3 ]
                pt2 = np.add( pt1, ray_dir( viewPoint, pt1 ) * dMax )
                pt3 = np.add( pt0, ray_dir( viewPoint, pt0 ) * dMax )
                surf.add_tri( pt0, pt1, pt2 )
                surf.add_tri( pt0, pt2, pt3 )
                farTri.append( pt2 )
            # Add the oposite triangle #
            surf.add_tri( farTri[2], farTri[1], farTri[0] )
            



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
