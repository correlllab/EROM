########## INIT ####################################################################################
import pickle, os, gc
from collections import deque
from copy import deepcopy
from pprint import pprint
from typing import Deque, Any

import numpy as np

from aspire.env_config import env_var
from aspire.symbols import GraspObj, ObjPose, euclidean_distance_between_symbols
from aspire.BlocksTask import set_blocks_env
from aspire.symbols import extract_position, extract_pose_as_homog

from TaskPlanner import set_experiment_env
from State import OCV_State_Tracker
from draw_beliefs import set_render_env
from draw_plots import make_histo, make_multi_histo

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
    if isinstance( sensedObjects, dict ):
        for lbl_i, obj_i in sensedObjects.items():
            matches[ id( obj_i ) ] = { "sensed" : obj_i, "known" : None, "d" : 6e10 }
    elif isinstance( sensedObjects, (list,deque,) ):
        for obj_i in sensedObjects:
            matches[ id( obj_i ) ] = { "sensed" : obj_i, "known" : None, "d" : 6e10 }
    else:
        raise ValueError( f"Could not parse a list of objects of type {type(sensedObjects)}" )
    # print( f"There are {len(matches)} symbols to match!" )
    
    # Match OpenCV Objects to Sensed Objects #
    for obj_j in lastScen:
        dMin   = 6e10
        kMin_i = None
        for k_i, v_i in matches.items():
            d_ij = euclidean_distance_between_symbols( v_i["sensed"], obj_j )
            # print( v_i["sensed"], obj_j )
            # print( f"d_ij: {d_ij}" )
            if d_ij is None:
                raise ValueError( f"Cannot compute a distance between {type(v_i['sensed'])} and {type(obj_j)}" )
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


def print_header( text : str, preWidth : int, totWidth : int, capitalize = True, _HDR_CHR : str = '#' ):
    """ Print a pleasant header """
    if capitalize:
        text = f"{text}".upper()
    totStr = '\n'*int(totWidth/25) + f"{preWidth*_HDR_CHR[0]} {text} "
    pstStr = max( totWidth-len(totStr)+1, 0 )*_HDR_CHR[0]
    if not len( pstStr ):
        pstStr = f"{preWidth*_HDR_CHR[0]}"
    totStr += pstStr
    print( totStr )



########## GATHER DATA #############################################################################
_MIN_STATE_SIZE_BYTES = 500.0

totRes : dict[str,dict] = dict()

fileDex = -1
banDex  = [70,79,80,93,95,96,142,144,146,147,148,149,150,151,152,153,154,155,156,157,158,159,161,]


try:
    ### For every block set ###
    for iii, paths in enumerate( datasets ):
        
        setNam = dataLabels[iii]
        suffix = "_" + setNam
        skip   = False

        totRes[ setNam ] = dict()

        ### For every scenario ###
        for ii, test in enumerate( tests ):
            
            print_header( f"TEST, {setNam}: {test}", preWidth = 10, totWidth = 100, capitalize = True )
            ##### Init ####################################################
            path     = paths[ii]
            longTNam = longTestNames[ii]
            
            testRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" not in f"{item}"))]
            trueRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" in f"{item}"))    ]

            def dex_key( x ):
                dex = f"{x}".split('_')[-1].replace( ".pkl", "" )
                if len( dex ) >= 2:
                    return dex
                elif len( dex ) < 2:
                    return '0'*(2-len( dex )) + dex
                else:
                    raise ValueError( "`dex_key`: This should NOT have happened!" )
                

            ##### Episode Basics ##########################################

            ### Episode Accounting ###
            results = {
                ## Steps ##
                "Nstep"   : deque(),
                "tStep"   : deque(),
                "tEpisd"  : deque(), # Total Makespan [s]
                "rSuccess": deque(), # Success Rate
                ## 2. Symbol Grounding ##
                "rConfuse" : deque(),
                "rFindFail": deque(),
            }

            ### For every episode ###
            for episodePath in testRecord:
                data = None 
                print( f"\n\nGarbage collector: Collected {gc.collect()} objects!" )
                fileDex += 1
                print_header( f"EP: {fileDex}, {episodePath}", preWidth = 5, totWidth = 75, capitalize = False )
                epPrefix = f"{episodePath}".replace( ".pkl", "" )
                

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

                ##### Per-Episode Accounting ##################################
                ### Steps ###
                Nstep    = 0
                tStepBgn = 0
                tStepEnd = 0
                tStepDqu = deque()
                ### 2. Symbol Grounding ###
                Nground    = 0
                totFound   = 0
                totConfuse = 0

                ##### Per-Message Accounting #####
                for datum in data:
                    dtmMsg = datum['msg']
                    dtmT   = datum['t']
                    dtmDat = datum['data']

                    ##### Phase 1: Perception #############################
                    if "BGN: Phase 1" in dtmMsg:
                        Nstep += 1
                        if tStepBgn > 0:
                            tStepEnd = dtmT
                            tStepDqu.append( tStepEnd - tStepBgn )
                        tStepBgn = dtmT

                ### Steps ###
                results["Nstep"].append( Nstep )
                results["tStep"].extend( tStepDqu )

                # DONE w `data`
                data = None 


                ##### Perception Metrics -vs- Ground Truth ################

                # WARNING, HACK: SKIP OVER FILES WITH PERCEPTION ISSUES
                if fileDex in banDex:
                    continue

                statePaths = [item for item in trueRecord if ((epPrefix in f"{item}") and ("_OCV-State" in f"{item}") and (os.path.getsize(item) >= _MIN_STATE_SIZE_BYTES))    ]
                statePaths.sort( key = lambda x: dex_key( x ) )
                statePaths = deque( statePaths )
                
                ### Load States ###
                _STATE_RAM_LIMIT_MB = 16e3
                state  = deque()
                sRAMmb = deque()
                sIndex = 0
                gotNum = False

                def pop_state():
                    """ Fetch next state """
                    sRAMmb.popleft()
                    return state.popleft()
                
                def p_numbered( fName : str ):
                    lastTwo = fName.split('.')[0][-1:]
                    try:
                        int( lastTwo )
                        return True
                    except ValueError:
                        return False
                    
                

                while len( statePaths ) or len( state ):

                    # 1. Load until limit 
                    while sum( sRAMmb ) < _STATE_RAM_LIMIT_MB:
                        if len( statePaths ):
                            sPath  = statePaths.popleft()
                            gotNum = gotNum or p_numbered( sPath )
                            if gotNum and (not p_numbered( sPath )):
                                continue
                            pathSz = int(os.path.getsize(sPath)/1e6)
                            try:
                                print( f"\t{sPath}, {pathSz}MB" )
                                with open( sPath, 'rb' ) as f:
                                    state.append( pickle.load(f) )
                                    sRAMmb.append( pathSz )
                            except Exception as e:
                                print( f">>>> SKIP {sPath}: {e} >>>>" )
                                continue
                        else:
                            break
                    print( f"Loaded {len( state )} states!" )

                    ### For every state ###
                    

                    # print( list( s_j.keys() ) ) # ['labels', 'image', 'depth', 'clouds', 'objects', 'symbols']
                    # "objects": deque(), # Collection of readings obtained from the masked images
                    # "symbols": dict(), #- Lookup of objects obtained from the readings
                    s_i = pop_state()
                    if not isinstance( s_i, (list,deque,) ):
                        s_i = [s_i,]
                    for s_j in s_i:
                        sense  = s_j['symbols']
                        print( f"Symbols: {sense}" )
                        truth  = reconcile_scene( s_j['objects'] )
                        print( f"Objects: {truth}" )
                        if not (len( sense ) or len( truth )):
                            continue
                        print_header( f"State {sIndex+1}", preWidth = 5, totWidth = 50, capitalize = False )
                        sIndex += 1
                        
                        conf_j = current_scene_confusion( sense, truth )
                        totFound   += conf_j['N_sensed' ]
                        totConfuse += conf_j['N_confuse']
                        Nground    += conf_j['N_true'   ]
                        print( conf_j )
                        sense = None
                        truth = None
                state = None
                    
                ### 2. Symbol Grounding ###
                results["rGround"].append( Nground / Nstep )
                results["rConfuse"].append( totConfuse / totFound )
                results["rFindFail"].append( (Nground - totFound) / Nground )

            totRes[ setNam ][ test ] = results
            # pprint( totRes )
except (KeyboardInterrupt, IndexError,):
    print( "\n\nSESSION CLOSED BY USER!" )
    crash_out()



########## MAKE PLOTS ##############################################################################

### Per color scenario ... ###
for scenario, scenDct in totRes.items():
    mSeries = deque()
    sNames  = deque()

    ##### Makespan ########################################################
    # {'RGB': {'KC-KP': 'tEpisd': deque([ ...

    ##### Makespan [Time] ######################## 
    for setting, stnDct in scenDct.items():
        mSeries.append( stnDct['tEpisd'] )
        sNames.append(  setting )
    make_multi_histo( mSeries, sNames, 
                      plotTitle = f"{datNamLong[ scenario ]}, {setting}\nMakespan Distribution [Time]", 
                      fName     = f"{_PLOT_DIR}Histo-Time_{scenario}{plotExt}", 
                      xLabel    = 'Time [s]', 
                      forceYlim = True, savefig = True )
    

    ##### Makespan [Steps] #######################
    for setting, stnDct in scenDct.items():
        mSeries.append( stnDct['Nstep'] )
        sNames.append(  setting )
    make_multi_histo( mSeries, sNames, 
                      plotTitle = f"{datNamLong[ scenario ]}, {setting}\nMakespan Distribution [Steps]", 
                      fName     = f"{_PLOT_DIR}Histo-Step_{scenario}{plotExt}", 
                      xLabel    = 'Steps', 
                      forceYlim = True, savefig = True )
    

    ##### Confusion Rate ##################################################

    ##### Makespan [Steps] #######################
    for setting, stnDct in scenDct.items():
        mSeries.append( stnDct['rConfuse'] )
        sNames.append(  setting )
    make_multi_histo( mSeries, sNames, 
                      plotTitle = f"{datNamLong[ scenario ]}, {setting}\nConfusion Rate", 
                      fName     = f"{_PLOT_DIR}Histo-Conf_{scenario}{plotExt}", 
                      xLabel    = 'Confusion Rate', 
                      forceYlim = True, savefig = True )



########## EXIT ####################################################################################
crash_out()