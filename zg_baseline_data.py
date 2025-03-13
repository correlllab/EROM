########## INIT && LOAD DATA #######################################################################
import pickle, os
from collections import deque

import numpy as np

from aspire.symbols import GraspObj
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import set_experiment_env
from draw_beliefs import ( set_render_env, render_memory_list, scan_geo, vispy_geo_list_window, cpcd_geo )

path = "data/Baseline_2025-03-11"
pkls = [os.path.join( path, item ) for item in os.listdir( path ) if ".pkl" in f"{item}".lower()]
for pkl in pkls:
    print( pkl )

dPth = pkls[0]

# print( f"About to open {dPth} ..." )

data = list()
with open( dPth, 'rb' ) as f:
    data = pickle.load( f )


##### Environment && Constants ############################################

set_blocks_env()
set_experiment_env()
set_render_env()


########## HELPER FUNCTIONS ########################################################################

def get_symbol_parents( objLst : list[GraspObj] ):
    """ Filter out the objects that are parents of symbols """
    rtnLst = list()
    for obj in objLst:
        if obj.SYM:
            rtnLst.append( obj )
    return rtnLst



########## ANALYSIS ################################################################################

_SUCCESS_RATE = False

##### Success Rate ########################################################
if _SUCCESS_RATE:
        N   = 0
        S   = 0
        F   = 0
        MTS = 0.0
        MTF = 0.0
        for pkl in pkls:
            print( f"About to open {pkl} ..." )
            with open( pkl, 'rb' ) as f:
                N   += 1
                data = pickle.load( f )
                msg  = data[-1]['msg']
                tRun = data[-1]['t'] - data[0]['t']
                print( msg )
                if "Status.FAILURE" in msg:
                    F += 1
                    print( "FAILURE" )
                    MTF += tRun
                elif "Status.SUCCESS" in msg:
                    S += 1
                    print( "SUCCESS" )
                    MTS += tRun
        MTS /= S
        MTF /= F
        print( f"\n{N} episodes, Success Rate: {S*1.0/N}, Failure Rate: {F*1.0/N}, Sanity Check == 0.0: {1.0-S*1.0/N-F*1.0/N}" )
        print( f"Mean Time to Success: {divmod( MTS, 60.0 )}, Mean Time to Failure: {divmod( MTF, 60.0 )}" )




########## DRAW SCANS ##############################################################################

totMem  = list()
camPose = np.eye(4)

_FETCH_CAM = False
_DRAW_SYMB = True
_DRAW_PCDS = True

dPth = pkls[-2]

print( f"About to open {dPth} ..." )

data = list()
with open( dPth, 'rb' ) as f:
    data = pickle.load( f )

for i, datum in enumerate( data ):

    if ((_FETCH_CAM or _DRAW_PCDS) and (datum['msg'] == 'camera')):
        camPose = datum['data'].copy()
        # print( datum['data'].keys() )

    if _DRAW_PCDS and (datum['msg'] == 'memory'):
        readings = datum['data']['scan']
        totGeo   = deque()
        for rdg in readings:
            if len( rdg.cpcd ) > 20:
                totGeo.extend( cpcd_geo( rdg ) )
                totGeo.extend( scan_geo( rdg ) )

        # print( type( totGeo ) )
        vispy_geo_list_window( list( totGeo ) )


        
    if _DRAW_SYMB and (datum['msg'] == 'symbols'):
        # render_memory_list( syms = datum['data']['scan'] )
        render_memory_list( syms = datum['data'] )




os.system( 'kill %d' % os.getpid() ) 