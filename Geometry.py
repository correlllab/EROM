########## INIT ####################################################################################

##### Imports #####

### Standard ###
from collections import deque

### Special ###
import numpy as np

### Local ###
from magpie_control.poses import vec_unit



########## GEOMETRY HELPERS ########################################################################

def tri_normal( p0, p1, p2 ):
    """ Return the unit normal vector for a triangle with points specified in CCW order """
    vec1 = np.subtract( p1 , p0 )
    vec2 = np.subtract( p2 , p0 )
    return vec_unit( np.cross( vec1 , vec2 ) )


def facet_adjacency_list_ordered( F ) -> list[list[int]]:
    """ Given a facet vertex lookup matrix 'F' , find all of the side-sharing neighbors of each facet , ORDERED VERSION """
    indices_RH = [ ( 0 , 1 ) , ( 1 , 2 ) , ( 2 , 0 ) ] # Right-hand indices
    indices_LH = [ ( 1 , 0 ) , ( 2 , 1 ) , ( 0 , 2 ) ] # Left-hand  indices , Neighbor will see vertices of the border segment in reverse order
    N = len( F ) # Number of facets
    neighborList = [ [ None , None , None ] for i in range( N ) ] # Ordered list of neighbors ( neighbor pointers )
    
    for i in range( N - 1 ):
        # Copy right-hand edges for each neighbor position that has not yet been associated with a neighbor
        neighbors = [ indices_RH[ checkDex ] if neighborList[i][ checkDex ] == None else None for checkDex in range(3) ] # Avoid repeat search
        for j in range( i + 1 , N ): # For each unique pairing of facets ( i , j )
            for nDex , neighbor in enumerate( neighbors ):
                if neighbor != None: # If we have not yet located the neighbor for this edge , Check needed to avoid repeats , see above
                    for nnDex , neighborNeighbor in enumerate( indices_LH ):
                        # nDex : This facet's edge  ,  nnDex : Candidate neighbor edge
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
        self.V = np.array( V, dtype = float ) if (V is not None) else np.zeros( (0,3), dtype = float )
        self.F = np.array( F, dtype = int   ) if (F is not None) else np.zeros( (0,3), dtype = int   )
        self.N = np.array( N, dtype = float ) if (N is not None) else np.zeros( (0,3), dtype = float )
        
    
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


    def get_raw_edges( self ):
        """ Get a list of any exposed edges in a non-enclosed mesh """
        rtnEdges = deque()
        adjacent = facet_adjacency_list_ordered( self.F )
        facDices = [ ( 0 , 1 ) , ( 1 , 2 ) , ( 2 , 0 ) ] # Right-hand indices
        for i, neighbors_i in enumerate( adjacent ):
            for nghbr_j in neighbors_i:
                if nghbr_j is None:
                    rtnEdges.append( [
                        self.V[ self.F[ i, facDices[0] ], : ],
                        self.V[ self.F[ i, facDices[1] ], : ]
                    ] )
        return list( rtnEdges )


    def get_occlusion_frustum( self, viewPoint : np.ndarray ):
        """ Get a region blocked by this object from the given viewpoint """
        surf = self.get_visible_submesh( viewPoint )