########## INIT ####################################################################################

import time
now = time.time
from random import choice

import numpy as np

from aspire.env_config import env_var
from aspire.utils import match_name, normalize_dist
from aspire.symbols import ( ObjPose, GraspObj, )

### Local ###
from utils import ( snap_z_to_nearest_block_unit_above_zero, LogPickler, 
                    zip_dict_sorted_by_decreasing_value, deep_copy_memory_list, )



########## HELPER FUNCTIONS ########################################################################


def hacked_offset_map( pose ) -> np.ndarray:
    """ Calculate a hack to the pose """
    hackXfrm = np.eye(4)
    offset   = np.zeros( (3,) )
    vec      = pose[0:3,3]
    
    minX     = env_var("_MIN_X_OFFSET")
    midX     = env_var("_MIN_X_OFFSET") + env_var("_X_WRK_SPAN")*0.50
    maxX     = env_var("_MAX_X_OFFSET")
    
    minY     = env_var("_MIN_Y_OFFSET")
    midY     = env_var("_MIN_Y_OFFSET") + env_var("_Y_WRK_SPAN")*0.50
    maxY     = env_var("_MAX_Y_OFFSET")

    height   = 0.5*env_var("_BLOCK_SCALE")

    hackMap  = [ [[minX, minY, height], [ 1.0/100.0, 1.0/100.0, 0.0]],
                 [[minX, maxY, height], [ 1.0/100.0, 0.0/100.0, 0.0]],
                 [[midX, midY, height], [ 1.0/100.0, 1.0/100.0, 0.0]],
                 [[maxX, minY, height], [ 1.0/100.0, 1.0/100.0, 0.0]], 
                 [[maxX, maxY, height], [ 1.0/100.0, 0.0/100.0, 0.0]],]
    
    weights = list()
    for hack in hackMap:
        weights.append( 1.0 / np.linalg.norm( np.subtract( vec, hack[0] ) ) )
    tot = sum( weights )
    for i, hack in enumerate( hackMap ):
        offset += (weights[i]/tot) * np.array( hack[1] )
    hackXfrm[0:3,3] = offset
    return hackXfrm


def observation_to_readings( obs : dict, xform = None ):
    """ Parse the Perception Process output struct """
    rtnBel = []
    if xform is None:
        xform = np.eye(4)
    for item in obs.values():
        dstrb = {}
        tScan = item['Time']

        # WARNING: CLASSES WITH A ZERO PRIOR WILL NOT ACCUMULATE EVIDENCE!
        for nam, prb in item['Probability'].items():
            if prb > 0.0001:
                dstrb[ match_name( nam ) ] = prb
            else:
                dstrb[ match_name( nam ) ] = env_var("_CONFUSE_PROB")
        if env_var("_NULL_NAME") not in dstrb:
            dstrb[ env_var("_NULL_NAME") ] = env_var("_CONFUSE_PROB")
        dstrb = normalize_dist( dstrb )

        if len( item['Pose'] ) == 16:
            # HACK: THERE IS A PERSISTENT GRASP OFFSET IN THE SCENE
            hackXfrm = hacked_offset_map( xform.dot( np.array( item['Pose'] ).reshape( (4,4,) ) )  )
            xform    = hackXfrm.dot( xform ) #env_var("_HACKED_OFFSET").dot( xform )
            
            objPose  = xform.dot( np.array( item['Pose'] ).reshape( (4,4,) ) ) 
            
            # HACK: SNAP TO NEAREST BLOCK UNIT && SNAP ABOVE TABLE
            objPose[2,3] = snap_z_to_nearest_block_unit_above_zero( objPose[2,3] )
        else:
            raise ValueError( f"`observation_to_readings`: BAD POSE FORMAT!\n{item['Pose']}" )
        
        # Create reading
        rtnObj = GraspObj( 
            labels = dstrb, 
            pose   = ObjPose( objPose ), 
            ts     = tScan, 
            count  = item['Count'], 
            score  = 0.0,
            cpcd   = item['CPCD'],
        )

        rtnBel.append( rtnObj )
    return rtnBel


def random_symbols_from_readings( objLst : list[GraspObj], N : int ):
    """ Randomly pick `N` readings to serve as symbols """
    if len( objLst ) < N:
        return list()
    picked = set([])
    rtnSym = list()
    while len( rtnSym ) < N:
        sym = choice( objLst )
        sID = id( sym )
        if sID not in picked:
            picked.add( sID )
            rtnSym.append( sym )
    for s in rtnSym:
        probs = zip_dict_sorted_by_decreasing_value( s.labels )
        s.label = probs[0][0]
        s.prob  = probs[0][1]
    return [sym.copy_child() for sym in rtnSym]
        



########## OBJECT PERMANENCE #######################################################################

### TBD ###



########## OBJECT MEMORY ###########################################################################

class Memory:
    """ Object Memory """

    def reset_memory( self ):
        """ Erase memory components """
        self.scan : list[GraspObj]  = list()


    def __init__( self ):
        self.history = LogPickler( prefix = "EROM-Memories", outDir = "data" )
        self.reset_memory()


    def shutdown( self ):
        """ Save the memory """
        self.history.dump_to_file( openNext = False )


    def process_observations( self, obs, xform = None ):
        """ Integrate one noisy scan into the current beliefs """
        self.scan = observation_to_readings( obs, xform )

        self.history.append( 
            datum = {
                "scan": deep_copy_memory_list( self.scan ),
            },
            msg = "memory" 
        )


    def get_current_most_likely( self ):
        """ Generate symbols """
        return random_symbols_from_readings( self.scan, env_var("_N_REQD_OBJS") )
        


    
    
    
    