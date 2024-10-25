########## INIT ####################################################################################

from math import isnan

import numpy as np

from aspire.symbols import GraspObj
from aspire.env_config import env_var



########## MEMORY FUNCTIONS ########################################################################

def copy_as_LKG( sym : GraspObj ):
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


def set_quality_score( obj : GraspObj ):
    """ Calc the score for this `GraspObj` """
    score_i = (1.0 - entropy_factor( obj.labels )) * obj.count
    if isnan( score_i ):
        print( f"\nWARN: Got a NaN score with count {obj.count} and distribution {obj.labels}\n" )
        score_i = 0.0
    obj.score = score_i


def snap_z_to_nearest_block_unit_above_zero( z : float ):
    """ SNAP TO NEAREST BLOCK UNIT && SNAP ABOVE TABLE """
    sHalf = (env_var("_BLOCK_SCALE")/2.0)
    zUnit = np.rint( (z-sHalf+env_var("_Z_SNAP_BOOST")) / env_var("_BLOCK_SCALE") ) # Quantize to multiple of block unit length
    zBloc = max( (zUnit*env_var("_BLOCK_SCALE"))+sHalf, sHalf )
    return zBloc