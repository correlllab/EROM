########## INIT && LOAD DATA #######################################################################
import pickle, os
from collections import deque

import numpy as np

from aspire.symbols import GraspObj, euclidean_distance_between_symbols
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

"""
* `[ ]` What if a reading overlaps with more than one reading: Which should it contribute to?
* `[ ]` Does the merge process make sense?

"""

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
                # print( list( tData.keys() ) ) # `dict`

                inpt = tData['input']
                print( f"input: {list(inpt.keys())}" ) # `dict`
                for k, v in inpt.items(): # ['query', 'abbrv', 'image', 'depth', 't']
                    print( f"\t{list(v.keys())}" )
                    
                hits = tData['hits']
                print( f"hits: {type(hits)}" ) # `list`
                print( f"\t{list(hits[0].keys())}" ) # ['bbox', 'bboxi', 'score', 'label', 'image', 'query', 'abbrv', 'shotID']

                print()


            elif tMsg == "memory":
                # print( list( tData.keys() ) ) # `dict`

                scan = tData['scan']
                print( f"scan: {type(scan)}" ) # `list`
                print( f"\t{scan[0]}" ) # `GraspObj`
                Mdst = np.zeros( (len(scan),len(scan),) )
                for i, obj_i in enumerate( scan ):
                    for j, obj_j in enumerate( scan ):
                        if i != j:
                            Mdst[i,j] = euclidean_distance_between_symbols( obj_i, obj_j )
                
                blfs = tData['beliefs']
                print( f"beliefs: {type(blfs)}" ) # `list`
                print( f"\t{blfs[0]}" ) # `GraspObj`
                Ndst = np.zeros( (len(blfs),len(scan),) )
                for i, obj_i in enumerate( blfs ):
                    for j, obj_j in enumerate( scan ):
                        Ndst[i,j] = euclidean_distance_between_symbols( obj_i, obj_j )

                print( Ndst )
                
                print()


            elif tMsg == "symbols":
                print( type( tData ) ) # `list``
                print()
            



########## CLEAN-UP ################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 
