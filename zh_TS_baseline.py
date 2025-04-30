########## INIT && LOAD DATA #######################################################################
import pickle, os
from collections import deque

import numpy as np

from aspire.env_config import env_var
from aspire.symbols import GraspObj, euclidean_distance_between_symbols
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import set_experiment_env
from draw_beliefs import ( set_render_env, render_scan_list, render_memory_list )

from Memory import Memory

# path = "data/Baseline_2025-03-11"
# path = "data/Baseline_2025-03-13" 
path = "data/Baseline_2025-04-28" 
# path = "data/Baseline_2025-04-29" 

pkls = [os.path.join( path, item ) for item in os.listdir( path ) if ".pkl" in f"{item}".lower()]
for pkl in pkls:
    print( pkl )
print()


##### Environment && Constants ############################################

set_blocks_env()
set_experiment_env()
set_render_env()



########## ANALYSIS ################################################################################
_TS_DETERM    = True
_TS_DETAIL    = True
_MEM_GRAPHICS = False

"""
- [>] ISSUE: Scan readings overlap a great deal, but these should have been merged during the Segmentation Phase!
    - [>] Log if segmentations are actually merged! (This can only be done in an experiment)
    - [ ] If they are not being merged, do they get properly integrated into the belief update?

- [>] ISSUE: The robot keeps missing the block!
    - [Y] How many stack actions end in failure?: 5 per run, NOT counting classification mistakes!
    - [>] Soln 1: One-shot sight-in, Based on the **closest** generic block
        - [>] Test Result: 
    - [P] Soln 2: Do not adjust poses of blocks placed by the robot
        - NEED TO THINK OF A WAY TO IMPLEMENT THIS THAT DOES NOT BREAK KL-DIVERGENCE TRACKING
        - [P] Test Result: 
    - [ ] Soln 3: Use the overhead camera
        - <+> An extra view without the wait
        - <-> Not setup for that, Get advice from Will

* [Y] What if a reading overlaps with more than one reading: Which should it contribute to?
    - A reading is already attributed to the belief that it is closest to
    - There isn't anything preventing more than one reading to contributing to the same belief
    - A single reading cannot contribute to multiple beliefs unless `integrate_one_to_many` is used
    
* [>] Does the belief update process make sense?
    - [Y] Check that massive AABB volume changes do not occur, 2025-04-25: Seems okay!
        `CPCD.merge()`: Volume changed by a factor of 1.11455
        `CPCD.merge()`: Volume changed by a factor of 1.05742
        `CPCD.merge()`: Volume changed by a factor of 1.00000
        `CPCD.merge()`: Volume changed by a factor of 1.00000
        `CPCD.merge()`: Volume changed by a factor of 1.00000
        `CPCD.merge()`: Volume changed by a factor of 1.00000

    - [>] Log the inputs of Bayes updates so you can see what is nudging them
    - [>] Log the readings that get eliminated && Visualize them

* [Y] Does the merge process make sense?, 2025-04-28: NO it did NOT!, This was a compound problem
    - Mask overlaps were insufficient on the segmentation side to create a distribution per pose
    - As a result, there were many overlaps in the initial scan
    - Planner had previously promoted the entire initial scan to beliefs, including overlaps
    - [P] What does it mean if a belief got an update before it was eliminated?
    - [P] Can an update make a belief more likely to be eliminated?
    - [P] What would happen if you did not do an update between scans?

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
    epRun = list()
    

    for pklDex, pklPath in enumerate( pkls ):
    # for pklDex, pklPath in enumerate( pkls[3:4] ):

        try:

            # We are going to troubleshoot how belief updates should go on the robot
            bMem   = Memory( None, None, suppressRecord = True ) 
            Nrun  += 1
            totRun = 0.0 
            totMty = 0.0

            print( f"Loading {pklPath} ...\n" )

            with open( pklPath, 'rb' ) as f:
                
                data    = pickle.load( f )
                totRun += (data[-1]['t'] - data[0]['t'])
                epRun.append( totRun )
                tLst    = data[0]['t']
                camPose = None

                for datum in data:
                    tMsg  = datum['msg']
                    tData = datum['data']
                    if _TS_DETAIL:
                        print( f"\n{datum['t']} : {tMsg}, {list(tData.keys()) if isinstance(tData,dict) else None}" )


                    if "Action Failure" in tMsg:
                        actFl += 1

                    if tMsg == "ObsMeta":

                        for k, v in tData.items():
                            print( f"{k}: ", end = "" )
                            if isinstance( v, dict ):
                                print( list( v.keys() ) )
                            elif isinstance( v, list ):
                                item = v[0]
                                if isinstance( item, dict ):
                                    print( list( item.keys() ) )
                            else:
                                print()

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

                    elif tMsg == 'camera':
                        camPose = datum['data'].copy()


                    elif tMsg == "memory":

                        scan = tData['scan']
                        print( list( tData.keys() ) )
                        
                        if _MEM_GRAPHICS:
                            render_scan_list( scan, camPose )

                        removed = bMem.process_observations( 
                            scan,
                            camPose,
                            False
                        ) 

                        if _MEM_GRAPHICS:
                            render_memory_list( bMem.bMem.beliefs, removed = removed )

                        if _TS_DETAIL:
                            print( f"scan: {type(scan)}" ) # `list`
                            # print( f"\t{scan[0]}" ) # `GraspObj`
                        Mdst = np.zeros( (len(scan),len(scan),) )
                        for i, obj_i in enumerate( scan ):
                            for j, obj_j in enumerate( scan ):
                                if i != j:
                                    Mdst[i,j] = euclidean_distance_between_symbols( obj_i, obj_j )
                        Mcls = np.where( Mdst < env_var("_BAYES_RAD_L2_M"), 1, 0)
                        
                        blfs = tData['beliefs']
                        if _TS_DETAIL:
                            print( f"beliefs: {type(blfs)}" ) # `list`
                            # print( f"\t{blfs[0]}" ) # `GraspObj`
                        Ndst = np.zeros( (len(blfs),len(scan),), dtype = int )
                        for i, obj_i in enumerate( blfs ):
                            for j, obj_j in enumerate( scan ):
                                Ndst[i,j] = euclidean_distance_between_symbols( obj_i, obj_j )
                        Ncls = np.where( Ndst < env_var("_BAYES_RAD_L2_M"), 1, Ndst )

                        if _TS_DETAIL:
                            print( f"\nThere are {len(blfs)} beliefs!\n" )
                            print( Mcls )
                            print()


                    elif tMsg == "symbols":

                        elapsed = datum['t'] - tLst
                        tLst    = datum['t']
                        Nsym    = len( tData )

                        if Nsym:
                            print( f"There are {Nsym} symbols!" )
                        else:
                            totMty += elapsed

                        # Reset memory every time we form a plan
                        bMem.reset_memory()

                        print()

                    elif tMsg == "Annotation":
                        print( f"{tData}\n" )

            if totMty > mxMty:
                mxMty = totMty
                mxIdx = pklDex
            mins, secs = divmod( totRun, 60.0 )
            mins = int( mins )
            print( f"\nSolver spent {totMty/totRun:.4f} of {mins}:{secs:.1f} without determinized symbols!" )  

        except KeyboardInterrupt:
            break
        
    epMin, epSec = [int(item) for item in divmod( sum(epRun)/Nrun, 60.0 )]
    mdMin, mdSec = [int(item) for item in divmod( np.median( epRun ), 60.0 )]
    mnMin, mnSec = [int(item) for item in divmod( min( epRun ), 60.0 )]
    mxMin, mxSec = [int(item) for item in divmod( max( epRun ), 60.0 )]
    print( f"\nAverage Running Time: {epMin}:{epSec}, Median Running Time: {mdMin}:{mdSec}\n" )
    print( f"\nMinimum Running Time: {mnMin}:{mnSec}, Maximum Running Time: {mxMin}:{mxSec}\n" )
    print( f"\nWorst run spent {mxMty} seconds without symbols!, Index: {mxIdx}" ) # Spends up to 1/3 of time without symbols!
    print( f"\nAverage Failed Actions per Run: {1.0*actFl/Nrun}" ) # Average Failed Actions per Run: 4.538
        

# 2025-04-29, Tue - Average Running Time: 12:17, Median Running Time: 9:25

########## CLEAN-UP ################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 
