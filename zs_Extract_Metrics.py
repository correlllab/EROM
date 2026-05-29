########## INIT ####################################################################################
import pickle, os, gc, traceback, json
from collections import deque
from copy import deepcopy
from pprint import pprint
from typing import Deque, Any

import numpy as np

from aspire.env_config import env_var
from aspire.symbols import GraspObj, ObjPose, euclidean_distance_between_symbols, p_symbol_inside_workspace_bounds
from aspire.BlocksTask import set_blocks_env
from aspire.symbols import extract_position, extract_pose_as_homog

from TaskPlanner import set_experiment_env
from State import OCV_State_Tracker
from draw_beliefs import set_render_env
from draw_plots import make_whisker, make_multi_histo


##### Environment && Constants ############################################
set_blocks_env()
set_experiment_env()
set_render_env()



########## SETUP ###################################################################################
_CONFUSION = False
_SAVE_THIN = False
_SAVE_DATA = True
_PLOT_DATA = False


# _DATA_DRIVE = "DATA_TANK"
_DATA_DRIVE = "STARGAZER/DATA_TANK"

# _PLOT_DIR   = "data/plots/"
_PLOT_DIR   = "/media/james/FILEPILE/EROM/data/plots/"
_GC_CYCLE   = False 
_F_EXTRACT  = f"{_PLOT_DIR}outData.pkl"
_T_EXTRACT  = f"{_PLOT_DIR}outText.json"

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

blcNam = {
    "RGB": ['redBlock','grnBlock','bluBlock',], 
    "RBW": ['redBlock','blkBlock','whtBlock',],
}

plotExt = ".pdf"



########## HELPER FUNCTIONS ########################################################################

def play_tone( duration_s = 5, freq_Hz = 650 ):
    """ Play a notification tone """
    os.system( f'play -nq -t alsa synth {duration_s} sine {freq_Hz}' )


def crash_out( notify = True ):
    """ End the program with Brutal Finality """
    if notify:
        play_tone()
    print( "\n\n" )
    os.system( 'kill %d' % os.getpid() ) 


def extract_label( obj : GraspObj ):
    """ Get most likely class """
    if (obj.label is None) or (obj.label == env_var("_NULL_NAME")):
        p = 0.0
        c = None
        for k, v in obj.labels.items():
            if v > p:
                p = v
                c = k
        return c
    else:
        return obj.label


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
    elif len( actualObjects ):
        pass
        # sensedObjects = most_likely_objects( sensedObjects, nameList = [item.label for item in actualObjects] )
    
    # Store Sensed Objects #
    if isinstance( sensedObjects, dict ):
        for lbl_i, obj_i in sensedObjects.items():
            matches[ id( obj_i ) ] = { "sensed" : obj_i, "known" : None, "d" : 6e10 }
    elif isinstance( sensedObjects, (list,deque,) ):
        for obj_i in sensedObjects:
            matches[ id( obj_i ) ] = { "sensed" : obj_i, "known" : None, "d" : 6e10 }
    else:
        raise ValueError( f"Could not parse a list of objects of type {type(sensedObjects)}" )
    
    # Match OpenCV Objects to Sensed Objects #
    usedSet = set([])
    for obj_j in lastScen:
        dMin   = 6e10
        kMin_i = None
        for k_i, v_i in matches.items():
            d_ij = euclidean_distance_between_symbols( v_i["sensed"], obj_j )
            if d_ij is None:
                raise ValueError( f"Cannot compute a distance between {type(v_i['sensed'])} and {type(obj_j)}" )
            if d_ij <= env_var("_ACCEPT_POSN_ERR") and d_ij < dMin:
                dMin   = d_ij
                kMin_i = k_i 
        if (kMin_i is not None) and (kMin_i not in usedSet):
            # print( f"d = {dMin} between {obj_j} and {matches[ kMin_i ]['sensed']}" )
            usedSet.add( kMin_i )
            matches[ kMin_i ]["known"] = obj_j
            matches[ kMin_i ]["d"    ] = dMin

    # Compute Number of Total, Confused, Hallucinated, and Missing Objects #
    Ncnf = 0 # Number of confusions
    Nhal = 0 # Number of hallucinations, False positive
    for k_i, v_i in matches.items():
        # If the sensed block was real, Then check for confusion
        if v_i["known"] is not None:
            # print(v_i["known"].label, v_i["sensed"].label)
            if extract_label( v_i["known"] ) != extract_label( v_i["sensed"] ):
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
    symbols = dict() # This is the final dictionary of symbols
    for obj_i in actualObjects:
        if not p_symbol_inside_workspace_bounds( obj_i, noPad = False, addMargin = 0.120 ):
            continue
        if len( obj_i.cpcd ):
            aabb = obj_i.cpcd.calc_aabb()
            if max(aabb[1,2], aabb[0,2]) < (0.25 * env_var("_BLOCK_SCALE")):
                continue
        lbl_i = obj_i.label
        # WARNING: THE FOLLOWING ASSUMES ONE OF EACH LABEL!
        if lbl_i in symbols:
            obj_j  = symbols[ lbl_i ]
            posn_i = extract_position( obj_i )
            posn_j = extract_position( obj_j )
            dst_ij = np.linalg.norm( np.subtract( posn_i, posn_j ) )
            factor = np.exp( -dst_ij*10 )*_BLEND_FACTOR
            posn_r = posn_i * factor + posn_j * (1.0 - factor)
            pose_r = extract_pose_as_homog( obj_j )
            pose_r[:3,3] = posn_r
            obj_j.pose = ObjPose( pose_r )
        else:
            symbols[ lbl_i ] = obj_i
    # print( f"Processed {len(actualObjects)} readings!" )
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


def copy_GraspObj_thin( objLst : list[GraspObj] ) -> list[GraspObj]:
    """ Copy a MUCH SMALLER version of the symbol! """
    rtnLst = deque()
    for obj in objLst:
        rtnLst.append( obj.copy( thin = True ) )
    return list( rtnLst )


def tokenize( expr : str ):
    """ Break a text `expr` into parts """
    _reserved = ['[', ']', ',',]
    expr += ' ' # Terminator hack
    token  = ""
    tokens = deque()

    def p_reserved( char ):
        return (char in _reserved)

    def store_token():
        nonlocal token, tokens
        if len( token ):
            try:
                tokens.append( float( token ) )
            except ValueError:
                tokens.append( token )
        token  = ""

    def store_char( char ):
        nonlocal tokens
        store_token()
        tokens.append( char )

    for char in expr:
        if char.isspace():
            store_token()
        elif p_reserved( char ):
            store_char( char )
        else:
            token += char

    return tokens


def extract_pose_from_tokens( tokens : list[str] ):
    """ Tokenize and parse a pose string """
    depth = 0
    matrx = deque()
    array = deque()
    for token in tokens:
        if token =='[':
            depth += 1
        if token ==']':
            depth -= 1
        if (depth == 2) and (not isinstance( token, str )):
            array.append( token )
        elif depth == 1:
            if len( array ):
                matrx.append( list( array ) )
                array = deque()
    if depth == 0:
        return np.array( list( matrx ) )
    else:
        return None
    

def extract_name_from_tokens( tokens : list[str] ):
    """ Get a block name from a list of tokens """
    for token in tokens:
        if 'Block' in token:
            return token
    return None


def get_name_and_origin( lines : list[str] ) -> np.ndarray:
    """ Get the origin and name of the block """
    accum  = False
    tokens = deque()
    name   = None
    pose   = None
    for line in lines:
        if ('Pick' in line) or ('Unstack' in line):
            accum = True
        if accum:
            linTkn = tokenize( line )
            tokens.extend( linTkn )
            name = extract_name_from_tokens( tokens )
            pose = extract_pose_from_tokens( tokens )
            if (name is not None) and (pose is not None):
                return name, pose
    return None, None


def get_desination( lines : list[str] ) -> np.ndarray:
    """ Get the origin and name of the block """
    accum  = False
    tokens = deque()
    pose   = None
    for line in lines:
        if ('Place' in line) or ('Stack' in line):
            accum = True
        if accum:
            linTkn = tokenize( line )
            tokens.extend( linTkn )
            pose = extract_pose_from_tokens( tokens )
            if (pose is not None):
                return pose
    return None


def parse_action( action : dict[str,list[str]] = None ):
    """ Get the intended class, origin, destination of the block """
    if action is not None:
        Lines    = action['next']
        dst      = get_desination( Lines )
        nam, src = get_name_and_origin( Lines )
    return {
        'name'   : nam,
        'bgnPose': src,
        'endPose': dst,
    }


def get_posn_variation( lastScene : list[GraspObj], thisScene : list[GraspObj], action = None ):
    """ Get a list of position variations between two scenes """
    namSet : set[str] = set([])
    if action is not None:
        action = parse_action( action )

    def exclude_source( scene : list[GraspObj], actDct : dict[str,str|np.ndarray] ):
        """ Exclude the location of the block we tried to move """
        nonlocal namSet
        try:
            # print( f"Try to ignore {actDct['name']} in {namSet}" )
            namSet.remove( actDct['name'] )
            # print( namSet )
        except KeyError:
            pass

    def get_block( blcLst : list[GraspObj], name : str ):
        """ Get a block from `blcLst` by `name` """
        for blc in blcLst:
            if extract_label( blc ) == name:
                return blc
        return None
    
    def get_names( scene : list[GraspObj] ):
        nonlocal namSet
        name = None
        for blc in scene:
            name = extract_label( blc )
            if name not in (None, env_var("_NULL_NAME"),):
                namSet.add( name )

    get_names( lastScene )
    get_names( thisScene )
    if action is not None:
        exclude_source( lastScene, action )

    varLst = deque()
    for name in namSet:
        lstBlc = get_block( lastScene, name )
        thsBlc = get_block( thisScene, name )
        if (None not in [lstBlc, thsBlc,]):
            d_n = euclidean_distance_between_symbols( lstBlc, thsBlc )
            # print( f"Distance: {d_n} b/n {lstBlc} and {thsBlc}" )
            varLst.append( d_n )
    return list( varLst )



########## WHAT IS GOING ON WITH CONFUSION? ########################################################
# banDex  = [70,79,80,93,95,96,142,144,146,147,148,149,150,151,152,153,154,155,156,157,158,159,161,]
banDex  = [72,142,]
fileDex = -1

if _CONFUSION:
    totRes = None
    try:
        with open( _F_EXTRACT, 'rb' ) as f:
            totRes = pickle.load(f)
    except Exception as e:
        traceback.print_exc()
        print( f"COULD NOT SAVE FILE: {e}" )
        crash_out()

    ### Per color scenario ... ###
    for scenario, scenDct in totRes.items():
        # {'RGB': {'KC-KP': 'tEpisd': deque([ ...
        for setting, stnDct in scenDct.items():
            print( f"\n\n########## {scenario}, {setting} ##########\n" )
            # pprint( stnDct ) # This is the `results` dict for each graph

            print( f"There are {len(stnDct['frames' ])} episodes to inspect" )
            print( f"There are {len(stnDct['actions'])} episodes to inspect" )
            
            for i, episode in enumerate( stnDct['frames'] ):
                print( f"\n##### {scenario}, {setting}, Ep. {i+1} #####" )
                actions   = stnDct['actions'][i]
                sense_jm1 = None
                truth_jm1 = None
                snsVar    = deque()
                truVar    = deque()
                print( f"There are {len(episode['states' ])} states to inspect" )
                print( f"There are {len(actions)} actions to inspect" )
                # print( f"There are {len(episode['actions'])} actions to inspect" )
                for j, state in enumerate( episode['states'] ):
                    jj      = j - 1
                    sense_j = state['sense']
                    truth_j = state['truth']
                    if j > 0:
                        action_j = actions[jj]
                        # pprint( action_j )
                        pprint( parse_action( action_j ) )
                        snsVar.extend( get_posn_variation( sense_j, sense_jm1, action = action_j ) )
                        truVar.extend( get_posn_variation( truth_j, truth_jm1, action = action_j ) )
                    sense_jm1 = sense_j
                    truth_jm1 = truth_j
                print( snsVar )
                print( truVar )
            crash_out( notify = False )



########## EXTRACT DATA FOR PLOTTING ###############################################################
if _SAVE_THIN:
    
    try:
        ### Open Data File ###
        totRes = None
        try:
            with open( _F_EXTRACT, 'rb' ) as f:
                totRes = pickle.load(f)
        except Exception as e:
            traceback.print_exc()
            print( f"COULD NOT SAVE FILE: {e}" )
            crash_out()

        ### Per color scenario ... ###
        for scenario, scenDct in totRes.items():
            # {'RGB': {'KC-KP': 'tEpisd': deque([ ...
            for setting, stnDct in scenDct.items():
                stnVar = {
                    'sensePosnVar' : deque(),
                    'truthPosnVar' : deque(),
                }
                print( f"\n\n########## {scenario}, {setting} ##########\n" )
                for i, episode in enumerate( stnDct['frames'] ):
                    print( f"\n##### {scenario}, {setting}, Ep. {i+1} #####" )
                    actions   = stnDct['actions'][i]
                    # print( f"Action Status: {stnDct['actStat']}" )

                    assert len( stnDct['actions'][i] ) == len( stnDct['actStat'][i] ), f"OH SHIT: {len( stnDct['actions'][i] )}, {len( stnDct['actStat'][i] )}"
                    print()
                    for j in range( len( stnDct['actions'][i] ) ):
                        print( stnDct['actStat'][i][j] )
                        pprint( parse_action( stnDct['actions'][i][j] ) )
                        print()

                    sense_jm1 = None
                    truth_jm1 = None
                    snsVar    = deque()
                    truVar    = deque()

                    print( f"There are {len(stnDct['actions'][i])} actions, {len(stnDct['actStat'][i])} statuses, and {len(episode['states'])} states!" )

                    for j, state in enumerate( episode['states'] ):
                        print( '>', end = '', flush = True )
                        jj      = j - 1
                        sense_j = state['sense']
                        truth_j = state['truth']
                        if (j > 0):
                            if (jj < len(stnDct['actions'][i])):
                                action_j = actions[jj]
                            else:
                                action_j = None
                            # pprint( parse_action( action_j ) )
                            snsVar.extend( get_posn_variation( sense_j, sense_jm1, action = action_j ) )
                            truVar.extend( get_posn_variation( truth_j, truth_jm1, action = action_j ) )
                        sense_jm1 = sense_j
                        truth_jm1 = truth_j
                    stnVar[ 'sensePosnVar' ].extend( snsVar )
                    stnVar[ 'truthPosnVar' ].extend( truVar )
                    print()
                totRes[ scenario ][ setting ][ 'posnVar' ] = {
                    'sense': deque( stnVar[ 'sensePosnVar' ] ),
                    'truth': deque( stnVar[ 'truthPosnVar' ] ),
                }

        ### Save Data File ###
        try:
            with open( _F_EXTRACT, 'wb' ) as f:
                pickle.dump( totRes, f )
        except Exception as e:
            traceback.print_exc()
            print( f"COULD NOT SAVE FILE: {e}" )
    except KeyboardInterrupt:
        print( "Session ENDED by the user!" )
                


########## GATHER DATA #############################################################################
_MIN_STATE_SIZE_BYTES = 500.0

totRes : dict[str,dict] = dict()
txtRes : dict[str,dict] = dict()

fileDex = -1

# banDex  = []

if _SAVE_DATA:
    try:
        ### For every block set ###
        for iii, paths in enumerate( datasets ):
            
            setNam = dataLabels[iii]
            suffix = "_" + setNam
            skip   = False

            totRes[ setNam ] = dict()
            txtRes[ setNam ] = dict()

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
                resText = {
                    ## Messages ##
                    "msgs": deque(),
                }
                results = {
                    ## Objects ##
                    "frames": deque(),
                    ## Steps ##
                    "Nstep"   : deque(),
                    "tStep"   : deque(),
                    "tEpisd"  : deque(), # Total Makespan [s]
                    "rSuccess": deque(), # Success Rate
                    ## 2. Symbol Grounding ##
                    "rConfuse" : deque(),
                    "rFindFail": deque(),
                    ### 3. Planning ###
                    "actions": deque(),
                    ### 4. Acting ###
                    "rActFail": deque(),
                    "actStat" : deque(),
                }

                ### For every episode ###
                for episodePath in testRecord:
                    eFrames = deque()
                    data    = None 
                    if _GC_CYCLE:
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
                    ### Messages ###
                    epMsgs = deque()
                    ### Steps ###
                    Nstep    = 0
                    tStepBgn = 0
                    tStepEnd = 0
                    tStepDqu = deque()
                    ### 2. Symbol Grounding ###
                    Nground    = 0
                    totFound   = 0
                    totConfuse = 0
                    jj         = 0
                    start      = False
                    added      = False
                    ### 3. Planning ###
                    actions = deque()
                    planned = False
                    ### 4. Acting ###
                    actStat = deque()
                    Naction   = 0
                    NfailActn = 0

                    ##### Per-Message Accounting #####
                    for datum in data:
                        epMsgs.append( datum['msg'] )
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
                            start    = True

                        ##### Phase 2: Grounding ##############################
                        if ("END: Phase 2" in dtmMsg) and start and len( dtmDat[:] ):
                            # estimates.append( dtmDat[:] )
                            start = False

                        if ("memory" in dtmMsg) and start and len( dtmDat["beliefs"] ):
                            # estimates.append( dtmDat["beliefs"] )
                            start = False

                        ##### Phase 3: Planning ###############################
                        if ("END: Phase 3" in dtmMsg):
                            actions.append( dtmDat )
                            if not len( dtmDat['next'] ):
                                actStat.append( None )
                            

                        ##### Phase 4: Execution ##############################
                        if "BGN: Phase 4" in dtmMsg:
                            Naction   += 1
                            actionFail = False

                        if ("BT END" in dtmMsg):
                            if ("fail" in f"{dtmMsg}".lower()):
                                NfailActn += 1
                                actionFail = True
                                actStat.append( False )
                            elif ("succ" in f"{dtmMsg}".lower()):
                                actStat.append( True )

                    resText["msgs"].append( epMsgs )
                    ### Steps ###
                    results["Nstep"].append( Nstep )
                    results["tStep"].extend( tStepDqu )
                    ### 4. Planning ###
                    results["actions"].append( list( actions ) )
                    ### 4. Acting ###
                    results["actStat" ].append( list( actStat ) )
                    results["rActFail"].append( NfailActn / Naction )

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
                        # "objects": deque(), # Collection of readings obtained from the masked images
                        # "symbols": dict(), #- Lookup of objects obtained from the readings
                        s_i = pop_state()
                        if not isinstance( s_i, (list,deque,) ):
                            s_i = [s_i,]
                        for s_j in s_i:
                            sense  = s_j['sensed']
                            print( f"Symbols: {sense}" )
                            truth  = list( s_j['symbols'].values() )
                            print( f"Objects: {truth}" )


                            eFrames.append({
                                'sense': copy_GraspObj_thin( sense ),
                                'truth': copy_GraspObj_thin( truth ),
                            })                            


                            if not len( truth ):
                                continue
                            print_header( f"State {sIndex+1}", preWidth = 5, totWidth = 50, capitalize = False )
                            sIndex += 1
                            # WARNING: THIS SMELLS
                            try:
                                # sense = estimates[jj]
                                jj   += 1
                                conf_j = current_scene_confusion( sense, truth )
                                totFound   += conf_j['N_sensed' ]
                                totConfuse += conf_j['N_confuse']
                                Nground    += conf_j['N_true'   ]
                                print( conf_j )
                            except IndexError:
                                pass
                            sense = None
                            truth = None
                    state = None
                    
                    results["frames"].append( {
                        'path'   : episodePath,
                        'setting': f"{setNam}, {test}",
                        'states' : list( eFrames )
                    } )
                    ### 2. Symbol Grounding ###
                    if totFound or Nground:
                        results["rConfuse"].append( totConfuse / Nground )
                        results["rFindFail"].append( (Nground - totFound) / Nground )

                totRes[ setNam ][ test ] = results
                txtRes[ setNam ][ test ] = resText
                # pprint( totRes )
    except (IndexError,):
        traceback.print_exc()
        crash_out()
    except (KeyboardInterrupt,):
        print( "\n\nSESSION CLOSED BY USER!" )
        crash_out()

    try:
        with open( _F_EXTRACT, 'wb' ) as f:
            pickle.dump( totRes, f )
        with open( _T_EXTRACT, 'w' ) as f:
            json.dump( txtRes, f, indent = 2 )
    except Exception as e:
        traceback.print_exc()
        print( f"COULD NOT SAVE FILE: {e}" )



########## MAKE PLOTS ##############################################################################

if _PLOT_DATA:

    try:
        with open( _F_EXTRACT, 'rb' ) as f:
            totRes = pickle.load( f )
            # pprint( totRes )
    except Exception as e:
        traceback.print_exc()
        print( f"COULD NOT SAVE FILE: {e}" )
        crash_out()

    trendData = {
        # What influence does Position Variation have on action failure?
        "Action Failure": deque(),
        "Position Var"  : deque(),
        # What influence do multiple blocks have on planning failure?
        "Planning Fail" : deque(),
        "Multiple Rate" : deque(),
    }

    ### Per color scenario ... ###
    for scenario, scenDct in totRes.items():
        
        ##### Makespan ########################################################
        # {'RGB': {'KC-KP': 'tEpisd': deque([ ...

        ##### Makespan [Time] ########################
        mSeries = deque()
        sNames  = deque()
        for setting, stnDct in scenDct.items():
            mSeries.append( stnDct['tEpisd'] )
            sNames.append(  setting )

            trendData["Position Var"  ].append( np.mean( stnDct['posnVar']['sense'] ) )
            trendData["Action Failure"].append( np.mean( stnDct['rActFail']         ) )


        make_multi_histo( mSeries, sNames, 
                          plotTitle = f"{datNamLong[ scenario ]}, Makespan Distribution [Time]", 
                          fName     = f"{_PLOT_DIR}Histo-Time_{scenario}{plotExt}", 
                          xLabel    = 'Time [s]', 
                          yLabel    = 'Occurrences',
                          forceYlim = True, savefig = True, decimals = 1 )
        make_whisker( mSeries, sNames, 
                      plotTitle = f"{datNamLong[ scenario ]}, Makespan Distribution [Time]", 
                      fName = f"{_PLOT_DIR}Whisker-Time_{scenario}{plotExt}", 
                      yLabel = "Seconds", 
                      forceYlim = False, savefig = True )
        

        ##### Makespan [Steps] #######################
        mSeries = deque()
        sNames  = deque()
        for setting, stnDct in scenDct.items():
            print( f"\n\n### {scenario}, {setting} ###\n" )
            if setting == "KC-KP":
                nStep = [elem for elem in stnDct['Nstep'] if (not (elem > 4))]
            else:
                nStep = list( stnDct['Nstep'] )
            mSeries.append( nStep   )
            sNames.append(  setting )
        make_multi_histo( mSeries, sNames, 
                          plotTitle = f"{datNamLong[ scenario ]}, Makespan Distribution [Steps]", 
                          fName     = f"{_PLOT_DIR}Histo-Step_{scenario}{plotExt}", 
                          xLabel    = 'Steps', 
                          yLabel    = 'Occurrences',
                          forceYlim = True, savefig = True, decimals = 0 )
        make_whisker( mSeries, sNames, 
                      plotTitle = f"{datNamLong[ scenario ]}, Makespan Distribution [Steps]", 
                      fName = f"{_PLOT_DIR}Whisker-Step_{scenario}{plotExt}", 
                      yLabel = "Steps", 
                      forceYlim = False, savefig = True )
        

        ##### Confusion / Perception Rates ####################################

        ##### Confusion ##############################
        mSeries = deque()
        sNames  = deque()
        for setting, stnDct in scenDct.items():
            print( f"\n\n### {scenario}, {setting} ###\n" )
            if setting == "KC-KP":
                cnfsn = [elem for elem in stnDct['rConfuse'] if (not (elem > 0.000005))]
            else:
                cnfsn = list( stnDct['rConfuse'] )
            mSeries.append( cnfsn )
            sNames.append(  setting )
        make_multi_histo( mSeries, sNames, 
                          plotTitle = f"{datNamLong[ scenario ]}, Confusion Rate", 
                          fName     = f"{_PLOT_DIR}Histo-Conf_{scenario}{plotExt}", 
                          xLabel    = 'Confusion Rate', 
                          yLabel    = 'Occurrences',
                          forceYlim = True, savefig = True, decimals = 3 )
        

        ##### Failure  Rates ####################################

        ##### Position Variation (Sense) #############
        mSeries = deque()
        sNames  = deque()
        for setting, stnDct in scenDct.items():
            # print( list( stnDct['posnVar'].keys() ) )
            mSeries.append( stnDct['posnVar']['sense'] )
            sNames.append(  setting )
        make_whisker( mSeries, sNames, 
                      plotTitle = f"{datNamLong[ scenario ]}, Variation of Sensed Position [m]", 
                      fName     = f"{_PLOT_DIR}Whisker-PosnVar-Sense_{scenario}{plotExt}", 
                      yLabel    = "[m]", 
                      forceYlim = False, 
                      savefig   = True,
                      outliers  = False )

        ##### Position Variation (Truth) #############
        mSeries = deque()
        sNames  = deque()
        for setting, stnDct in scenDct.items():
            mSeries.append( stnDct['posnVar']['truth'] )
            sNames.append(  setting )
        make_whisker( mSeries, sNames, 
                      plotTitle = f"{datNamLong[ scenario ]}, Variation of Ground Truth Position [m]", 
                      fName     = f"{_PLOT_DIR}Whisker-PosnVar-Truth_{scenario}{plotExt}", 
                      yLabel    = "[m]", 
                      forceYlim = False, 
                      savefig   = True, 
                      outliers  = False )
        



########## EXIT ####################################################################################
crash_out( notify = (_SAVE_DATA or _SAVE_THIN) )