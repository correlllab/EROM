########## INIT && LOAD DATA #######################################################################
import pickle, os
from collections import deque

import numpy as np

from aspire.env_config import env_var
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
_TS_DETAIL = True

"""
- [ ] ISSUE: The robot keeps missing the block!
    - [Y] How many stack actions end in failure?: 5 per run, NOT counting classification mistakes!
    - [>] Soln 1: One-shot sight-in, Based on the **closest** generic block
        - [>] Test Result: 
    - [ ] Soln 2: Do not asjust poses of blocks placed by the robot
        - [ ] Test Result: 

* [Y] What if a reading overlaps with more than one reading: Which should it contribute to?
    - A reading is already attributed to the belief that it is closest to
    - There isn't anything preventing more than one reading to contributing to the same belief
    - A single reading cannot contribute to multiple beliefs unless `integrate_one_to_many` is used

* [ ] Does the merge process make sense?
    - [ ] What does it mean if a belief got an update before it was eliminated?
    - [ ] Can an update make a belief more likely to be eliminated?
    - [ ] What would happen if you did not do an update between scans?

* [ ] What am I supposed to do when there are more beliefs than symbols?
    - [ ] Is there a way to consider overlapping symbols in the determination?

* [ ] Visualize
    - [ ] Which readings contribute to what beliefs?
    - [ ] Which beliefs got eliminated?
    - [ ] What were the distribution of the beliefs that got chosen?

"""

if _TS_DETERM:

    mxMty = 0.0
    mxIdx = -1
    actFl = 0
    Nrun  = 0

    for pklDex, pklPath in enumerate( pkls ):
    # for pklDex, pklPath in enumerate( pkls[3:4] ):

        Nrun  += 1
        totRun = 0.0 
        totMty = 0.0

        print( f"Loading {pklPath} ...\n" )

        with open( pklPath, 'rb' ) as f:
            
            data   = pickle.load( f )
            totRun += (data[-1]['t'] - data[0]['t'])
            tLst   = data[0]['t']

            for datum in data:
                tMsg  = datum['msg']
                tData = datum['data']
                if _TS_DETAIL:
                    print( f"\n{datum['t']} : {tMsg}" )


                if "Action Failure" in tMsg:
                    actFl += 1

                if tMsg == "ObsMeta":

                    inpt = tData['input']
                    if _TS_DETAIL:
                        print( f"input: {list(inpt.keys())}" ) # `dict`
                        for k, v in inpt.items(): # ['query', 'abbrv', 'image', 'depth', 't']
                            print( f"\t{list(v.keys())}" )
                        
                    hits = tData['hits']
                    if _TS_DETAIL:
                        print( f"hits: {type(hits)}" ) # `list`
                        print( f"\t{list(hits[0].keys())}" ) # ['bbox', 'bboxi', 'score', 'label', 'image', 'query', 'abbrv', 'shotID']
                        print()


                elif tMsg == "memory":

                    scan = tData['scan']
                    if _TS_DETAIL:
                        print( f"scan: {type(scan)}" ) # `list`
                        print( f"\t{scan[0]}" ) # `GraspObj`
                    Mdst = np.zeros( (len(scan),len(scan),) )
                    for i, obj_i in enumerate( scan ):
                        for j, obj_j in enumerate( scan ):
                            if i != j:
                                Mdst[i,j] = euclidean_distance_between_symbols( obj_i, obj_j )
                    Mcls = np.where( Mdst < env_var("_BAYES_RAD_L2_M"), 1, 0)
                    
                    blfs = tData['beliefs']
                    if _TS_DETAIL:
                        print( f"beliefs: {type(blfs)}" ) # `list`
                        print( f"\t{blfs[0]}" ) # `GraspObj`
                    Ndst = np.zeros( (len(blfs),len(scan),), dtype = int )
                    for i, obj_i in enumerate( blfs ):
                        for j, obj_j in enumerate( scan ):
                            Ndst[i,j] = euclidean_distance_between_symbols( obj_i, obj_j )
                    Ncls = np.where( Ndst < env_var("_BAYES_RAD_L2_M"), 1, Ndst)

                    if _TS_DETAIL:
                        print( f"\nThere are {len(blfs)} beliefs!\n" )
                        print( Mcls )
                        print()
                        # print( Ncls ) # This is always 1's!


                elif tMsg == "symbols":

                    elapsed = datum['t'] - tLst
                    tLst    = datum['t']
                    Nsym    = len( tData )

                    if Nsym:
                        print( f"There are {Nsym} symbols!" )
                    else:
                        totMty += elapsed

                    print()

                elif tMsg == "Annotation":
                    print( f"{tData}\n" )


        if totMty > mxMty:
            mxMty = totMty
            mxIdx = pklDex
        mins, secs = divmod( totRun, 60.0 )
        mins = int( mins )
        print( f"\nSolver spent {totMty/totRun:.4f} of {mins}:{secs:.1f} without determinized symbols!" )  
    
    print( f"\nWorst run spent {mxMty} seconds without symbols!, Index: {mxIdx}" ) # Spends up to 1/3 of time without symbols!
    print( f"\nAverage Failed Actions per Run: {1.0*actFl/Nrun}" ) # Average Failed Actions per Run: 4.538

# Solver spent 0.2068 of 353:12.0 without determinized symbols!

########## CLEAN-UP ################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 
