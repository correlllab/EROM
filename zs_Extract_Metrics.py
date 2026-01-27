########## INIT ####################################################################################
import pickle, os, traceback, json
from collections import deque
from copy import deepcopy
from pprint import pprint
from typing import Deque, Any

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
    [ f"/media/james/{_DATA_DRIVE}/RWB_2025-09_{test}" for test in tests ],
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


def current_scene_confusion( sensedObjects : list[GraspObj], actualObjects : list[GraspObj] ):
    """ Return the number of `sensedObjects` that *contradict* the current scene """
    lastScen = actualObjects
    matches : dict[int,dict[str,Any|GraspObj]]  = dict()
    
    if (sensedObjects is None) or (not len( sensedObjects )):
        return {
        "N_confuse": 0,
        "N_halluc" : 0,
        "N_missing": len( lastScen ),
        "N_sensed" : 0,
        "N_true"   : len( lastScen ),
    }
    
    # Store Sensed Objects #
    for obj_i in sensedObjects:
        matches[ id( obj_i ) ] = { "sensed" : obj_i, "known" : None, "d" : 6e10 }
    
    # Match OpenCV Objects to Sensed Objects #
    for obj_j in lastScen:
        dMin   = 6e10
        kMin_i = None
        for k_i, v_i in matches.items():
            d_ij = euclidean_distance_between_symbols( v_i["sensed"], obj_j )
            if d_ij is None:
                continue
            if d_ij <= env_var("_ACCEPT_POSN_ERR") and d_ij < dMin:
                dMin   = d_ij
                kMin_i = k_i 
        if kMin_i is not None:
            matches[ kMin_i ]["known"] = obj_j
            matches[ kMin_i ]["d"    ] = dMin
        
    # Compute Number of Total, Confused, Hallucinated, and Missing Objects #
    Ncnf = 0 # Number of confusions
    Nhal = 0 # Number of hallucinations, False positive
    for k_i, v_i in matches.items():
        # If the sensed block was real, Then check for confusion
        if v_i["known"] is not None:
            if v_i["known"].label != v_i["sensed"].label:
                Ncnf += 1
        # Else block was NOT real, The system hallucinated it! 
        else:
            Nhal += 1 
    Nmis = max( len( lastScen )-len( sensedObjects ), 0 )
    return {
        "N_confuse": Ncnf,
        "N_halluc" : Nhal,
        "N_missing": Nmis,
        "N_sensed" : len( sensedObjects ),
        "N_true"   : len( lastScen ),
    }


from State import OCV_State_Tracker
from aspire.symbols import extract_position, extract_pose_as_homog


def reconcile_scene( actualObjects : list[GraspObj] ):
    """ Merge all the readings """
    _BLEND_FACTOR = 0.200
    symbols = dict()
    for obj_i in actualObjects:
        lbl_i = obj_i.label
        # WARNING: THE FOLLOWING ASSUMES ONE OF EACH LABEL!
        if lbl_i in symbols:
            obj_j  = symbols[ lbl_i ]
            posn_i = extract_position( obj_i )
            posn_j = extract_position( obj_j )
            posn_r = posn_i * _BLEND_FACTOR + posn_j * (1.0 - _BLEND_FACTOR)
            pose_r = extract_pose_as_homog( obj_j )
            pose_r[:3,3] = posn_r
            obj_j.pose = ObjPose( pose_r )
        else:
            symbols[ lbl_i ] = obj_i
    print( f"Processed {len(actualObjects)} readings!" )
    rtnSym = list( symbols.values() )
    
    OCV_State_Tracker.logical_Z_snap( rtnSym )
    return rtnSym


########## SAVE: DATA PROCESSING ###################################################################
_SAVE_DATA      = True
_LOAD_DATA      = True 
_MIN_STATE_SIZE = 500.0

records = deque()
grTruth = deque()

def print_header( text : str, preWidth : int, totWidth : int, capitalize = True, _HDR_CHR : str = '#' ):
    """ Print a pleasant header """
    if capitalize:
        text = f"{text}".upper()
    totStr = f"\n{preWidth*_HDR_CHR[0]} {text} "
    totStr += max( totWidth-len(totStr)+1, 0 )*_HDR_CHR[0]
    print( totStr )


try:
    ### For every block set ###
    for iii, paths in enumerate( datasets ):
        
        setNam = dataLabels[iii]
        suffix = "_" + setNam
        skip   = False

        ### For every scenario ###
        for ii, test in enumerate( tests ):
            print_header( f"TEST: {test}", preWidth = 10, totWidth = 100, capitalize = True )
            ##### Init ####################################################
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
                

            ##### Episode Basics ##########################################

            ### For every episode ###
            for episodePath in testRecord:
                print_header( f"EP: {episodePath}", preWidth = 5, totWidth = 75, capitalize = False )
                epPrefix   = f"{episodePath}".replace( ".pkl", "" )
                statePaths = [item for item in trueRecord if ((epPrefix in f"{item}") and ("_OCV-State" in f"{item}") and (os.path.getsize(item) >= _MIN_STATE_SIZE))    ]
                statePaths.sort( key = lambda x: dex_key( x ) )

                ### Episode Accounting ###
                results = {
                    ## Steps ##
                    "tEpisd"  : deque(), # Total Makespan [s]
                    "rSuccess": deque(), # Success Rate
                }

                ### Makespan Metrics ###
                print( f"\n{episodePath}, {int(os.path.getsize(episodePath)/1e6)}MB" )
                try:
                    with open( episodePath, 'rb' ) as f:
                        data = pickle.load( f )
                except EOFError as e:
                    print( f"LOAD ERROR: {e}" )
                    continue

                end   =  False
                resEp = 0
                for i in range(1,11):
                    try:
                        msg = data[-i]['msg']
                    except IndexError as e:
                        print(e)
                        break
                    if ("Status.FAILURE" in msg) and ("BT END" not in msg) and ("Behavior" not in msg):
                        resEp = 0
                        print( f"FAILURE" )
                        end = True
                        break
                    elif ("Status.SUCCESS" in msg) and ("BT END" not in msg) and ("Behavior" not in msg):
                        resEp = 1
                        print( f"SUCCESS" )
                        end = True
                        break
                if not end:
                    resEp = 0
                    print( f"FAILURE" )

                results["tEpisd"].append( data[-1]['t'] - data[0]['t'] )
                results["rSuccess"].append( resEp )


                ##### Perception Metrics -vs- Ground Truth ################
                
                ### Load States ###
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

                ### For every state ###
                for j, s_j in enumerate( state ):
                    print( f"\n##### State {j+1} #####" )
                    # print( list( s_j.keys() ) ) # ['labels', 'image', 'depth', 'clouds', 'objects', 'symbols']
                    # "objects": deque(), # Collection of readings obtained from the masked images
                    # "symbols": dict(), #- Lookup of objects obtained from the readings
                    sense  = s_j['symbols']
                    truth  = reconcile_scene( s_j['objects'] )
                    conf_j = current_scene_confusion( sense, truth )
                    print( conf_j )


                
                    
                    
                    
                    

except KeyboardInterrupt:
    print( "\n\nSESSION CLOSED BY USER!" )
    crash_out()

########## EXIT ####################################################################################
crash_out()