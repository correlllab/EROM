########## INIT ####################################################################################
import pickle, os, traceback, json
from collections import deque
from copy import deepcopy
from pprint import pprint
from typing import Deque

import numpy as np

from aspire.env_config import env_var
from aspire.symbols import GraspObj, ObjPose, euclidean_distance_between_symbols
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import set_experiment_env
from draw_beliefs import set_render_env
from utils import deep_copy_memory_list

##### Environment && Constants ############################################
set_blocks_env()
set_experiment_env()
set_render_env()


########## SETUP ###################################################################################
_JSON_PATH  = "data/allData.txt"
_DATA_DRIVE = "DATA_TANK"
_PLOT_DIR   = "data/plots/"

tests = [
    "KC-KP",
    "SC-KP",
    "KC-SP",
    "SC-SP",
]

longTestNames = [
    "Known Class & Known Pose", 
    "Sensed Class & Known Pose", 
    "Known Class & Sensed Pose", 
    "Sensed Class & Sensed Pose", 
]

datasets = [
    [ f"/media/james/{_DATA_DRIVE}/2025-08B_{test}" for test in tests ],
    # [ f"/media/james/{_DATA_DRIVE}/RWB_2025-09_{test}" for test in tests ],
]

dataLabels = ["RGB", "RBW",]
datNamLong = {
    "RGB": "Red-Green-Blue", 
    "RBW": "Red-Black-White",
}

plotExt = ".pdf"


########## HELPER FUNCTIONS ########################################################################

def crash_out():
    """ End the program with Brutal Finality """
    print( "\n\n" )
    os.system( 'kill %d' % os.getpid() ) 



########## SAVE: DATA PROCESSING ###################################################################
_SAVE_DATA      = True
_LOAD_DATA      = True 
_MIN_STATE_SIZE = 500.0

records = deque()
grTruth = deque()

try:
    ### For every block set ###
    for iii, paths in enumerate( datasets ):
        setNam = dataLabels[iii]
        suffix = "_" + setNam
        skip   = False

        ### For every scenario ###
        for ii, test in enumerate( tests ):
            ##### Init ################################################################
            path     = paths[ii]
            longTNam = longTestNames[ii]
            
            testRecord = [os.path.join( path, item ) for item in os.listdir( path ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" not in f"{item}"))]
            trueRecord = [os.path.join( path, item ) for item in os.listdir( path ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" in f"{item}"))    ]

            def dex_key( x ):
                dex = f"{x}".split('_')[-1].replace( ".pkl", "" )
                if len( dex ) >= 2:
                    return dex
                elif len( dex ) < 2:
                    return '0'*(2-len( dex )) + dex
                else:
                    raise ValueError( "`dex_key`: This should NOT have happened!" )

            ### For every episode ###
            for episodePath in testRecord:
                epPrefix   = f"{episodePath}".replace( ".pkl", "" )
                statePaths = [item for item in trueRecord if ((epPrefix in f"{item}") and ("_OCV-State" in f"{item}") and (os.path.getsize(item) >= _MIN_STATE_SIZE))    ]
                statePaths.sort( key = lambda x: dex_key( x ) )

                print( f"\n{episodePath}, {int(os.path.getsize(episodePath)/1e6)}MB" )

                state = deque()
                if len( statePaths ) == 1:
                    print( f"\t{statePaths[0]}, {int(os.path.getsize(statePaths[0])/1e6)}MB" )
                    with open( statePaths[0], 'rb' ) as f:
                        state = pickle.load(f)
                elif len( statePaths ) > 1:
                    ### For every state file ###
                    for sPath in statePaths:
                        try:
                            print( f"\t{sPath}, {int(os.path.getsize(sPath)/1e6)}MB" )
                            with open( sPath, 'rb' ) as f:
                                state.append( pickle.load(f) )
                        except Exception as e:
                            print( f">>>> SKIP {sPath}: {e} >>>>" )
                            continue
                        if state is None:
                            print( f">>>> SKIP {sPath}: NONE STATE >>>>" )
                            continue
                        print( f"\t\t{type(state[-1])}" )
                else:
                    raise ValueError( "ZERO State Files!!" )
                print( f"Loaded {len( state )} states!" )

                ### Makespan Metrics ###


                ### For every state ###
                for j, s_j in enumerate( state ):
                    print( f"\n##### State {j+1} #####" )
                    # print( list( s_j.keys() ) ) # ['labels', 'image', 'depth', 'clouds', 'objects', 'symbols']
                    # "objects": deque(), # Collection of readings obtained from the masked images
                    # "symbols": dict(), #- Lookup of objects obtained from the readings
                    sense = s_j['symbols']
                    truth = s_j['objects']


                
                    
                    
                    
                    

except KeyboardInterrupt:
    print( "\n\nSESSION CLOSED BY USER!" )
    crash_out()

########## EXIT ####################################################################################
crash_out()