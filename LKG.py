from math import isnan

import numpy as np

from aspire.env_config import env_var
from aspire.symbols import GraspObj, ObjPose
from aspire.utils import match_name



########## MEMORY HELPER FUNCTIONS #################################################################

def copy_as_LKG( sym ):
    """ Make a copy of this belief for the Last-Known-Good collection """
    rtnObj = sym.copy()
    rtnObj.LKG = True
    return rtnObj

def copy_readings_as_LKG( readLst ):
    """ Return a list of readings intended for the Last-Known-Good collection """
    rtnLst = list()
    for r in readLst:
        rtnLst.append( copy_as_LKG( r ) )
    return rtnLst



########## PROBABILITY HELPER FUNCTIONS ############################################################

def entropy_factor( probs ):
    """ Return a version of Shannon entropy scaled to [0,1] """
    if isinstance( probs, dict ):
        probs = list( probs.values() )
    tot = 0.0
    # N   = 0
    for p in probs:
        pPos = max( p, 0.00001 )
        tot -= pPos * np.log( pPos )
            # N   += 1
    return tot / np.log( len( probs ) )



########## LAST KNOWN GOOD #########################################################################


class LKG:
    """ Last Known Good memory manager """

    def reset_state( self ):
        """ Erase memory components """
        self.scan   = list()
        self.memory = list()

    def __init__( self ):
        pass

    def snap_z_to_nearest_block_unit_above_zero( self, z ):
        """ SNAP TO NEAREST BLOCK UNIT && SNAP ABOVE TABLE """
        sHalf = (env_var("_BLOCK_SCALE")/2.0)
        zUnit = np.rint( (z-sHalf+env_var("_Z_SNAP_BOOST")) / env_var("_BLOCK_SCALE") ) # Quantize to multiple of block unit length
        zBloc = max( (zUnit*env_var("_BLOCK_SCALE"))+sHalf, sHalf )
        return zBloc
    
    def observation_to_readings( self, obs, xform = None ):
        """ Parse the Perception Process output struct """
        rtnBel = []
        if xform is None:
            xform = np.eye(4)
        for item in obs.values():
            dstrb = {}
            tScan = item['Time']

            for nam, prb in item['Probability'].items():
                dstrb[ match_name( nam ) ] = prb
            if env_var("_NULL_NAME") not in dstrb:
                dstrb[ env_var("_NULL_NAME") ] = 0.0

            if len( item['Pose'] ) == 16:
                objPose = xform.dot( np.array( item['Pose'] ).reshape( (4,4,) ) ) 
                # HACK: SNAP TO NEAREST BLOCK UNIT && SNAP ABOVE TABLE
                objPose[2,3] = self.snap_z_to_nearest_block_unit_above_zero( objPose[2,3] )
            else:
                raise ValueError( f"`observation_to_readings`: BAD POSE FORMAT!\n{item['Pose']}" )
            
            # Attempt to quantify how much we trust this reading
            score_i = (1.0 - entropy_factor( dstrb )) * item['Count']
            if isnan( score_i ):
                print( f"\nWARN: Got a NaN score with count {item['Count']} and distribution {dstrb}\n" )
                score_i = 0.0

            rtnBel.append( GraspObj( 
                labels = dstrb, pose = ObjPose( objPose ), ts = tScan, count = item['Count'], score = score_i 
            ) )
        return rtnBel
