########## INIT && LOAD DATA #######################################################################
import pickle, os
from collections import deque

import numpy as np

from aspire.symbols import GraspObj
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import set_experiment_env
from draw_beliefs import ( set_render_env, render_memory_list, scan_geo, vispy_geo_list_window, cpcd_geo )

# path = "data/Baseline_2025-03-11"
path = "data/Baseline_2025-03-13"

pkls = [os.path.join( path, item ) for item in os.listdir( path ) if ".pkl" in f"{item}".lower()]
for pkl in pkls:
    print( pkl )
print()


##### Environment && Constants ############################################

set_blocks_env()
set_experiment_env()
set_render_env()



########## ANALYSIS ################################################################################
_TS_DETERM = True

pklPath = pkls[0]

if _TS_DETERM:
    print( f"Loading {pklPath} ...\n" )
    with open( pklPath, 'rb' ) as f:
        data = pickle.load( f )
        for datum in data:
            print()
            tMsg  = datum['msg']
            tData = datum['data']
            print( f"{datum['t']} : {tMsg}" )
            if tMsg == "ObsMeta":
                print( type( tData ) ) # `dict`
                print()
            elif tMsg == "memory":
                print( type( tData ) ) # `dict`
                print()
            elif tMsg == "symbols":
                print( type( tData ) ) # `list``
                print()
            



########## CLEAN-UP ################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 
