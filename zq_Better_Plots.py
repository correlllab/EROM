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
# _PLOT_DIR   = "/media/james/FILEPILE/EROM/data/plots/"
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

# paths = [ f"/media/james/{_DATA_DRIVE}/2025-08_{test}" for test in tests ]
datasets = [
    [ f"/media/james/{_DATA_DRIVE}/2025-08B_{test}" for test in tests ],
    [ f"/media/james/{_DATA_DRIVE}/RWB_2025-09_{test}" for test in tests ],
]

dataLabels = ["RGB", "RBW",]
datNamLong = {
    "RGB": "Red-Green-Blue", 
    "RBW": "Red-Black-White",
}

fNames = [ f"{_PLOT_DIR}{test}" for test in tests  ]

plotExt = ".pdf"


########## HELPER FUNCTIONS ########################################################################
import matplotlib.pyplot as plt


_TITLE_FONT_SIZE = 13
_TITLE_SMOL_SIZE = 10
_N_TRIALS        = 20
_TIGHT_MARGIN    =  0.05


def make_histo( series, plotTitle, fName, xLabel = 'Makespan', yLabel = 'Occurrences', forceYlim = True, savefig = True ):
    """ Create Histogram """
    if savefig:
        plt.clf()
    plt.margins( _TIGHT_MARGIN )
    print( f"\n{plotTitle}" )
    print( f"Mean: ___ {np.mean(series)}" )
    print( f"Median: _ {np.median(series)}" )
    print( f"Std.Dev.: {np.std(series)}" )
    plt.hist( series )
    plt.title( plotTitle, fontsize = _TITLE_FONT_SIZE ) # Set the title && font size
    plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    if forceYlim:
        plt.ylim( (0, _N_TRIALS,) )
    plt.tight_layout()
    if savefig:
        plt.savefig( fName )
        return plt.gca()


def make_multi_histo( multiSeries, seriesNames, plotTitle = None, fName = "output.pdf", xLabel = None, yLabel = None, 
                      forceYlim = True, savefig = True, titleFontSize_pt = _TITLE_FONT_SIZE ):
    """ Create Histogram """
    if savefig:
        plt.clf()
    plt.margins( _TIGHT_MARGIN )
    print( f"\n### {plotTitle} ###" )
    for i, series in enumerate( multiSeries ):
        print( f"\t{seriesNames[i]}" )
        print( f"\tMean: ___ {np.mean(series)}" )
        print( f"\tMedian: _ {np.median(series)}" )
        print( f"\tStd.Dev.: {np.std(series)}" )
    plt.hist( multiSeries, label = seriesNames )
    if plotTitle is not None:
        plt.title( plotTitle, fontsize = titleFontSize_pt ) # Set the title && font size
    if xLabel is not None:
        plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    if yLabel is not None:
        plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    if savefig:
        plt.legend( loc = 'upper right' )
    if forceYlim:
        plt.ylim( (0, _N_TRIALS,) )
    plt.tight_layout()
    if savefig:
        plt.savefig( fName )
        return plt.gca()


def p_str_has_any( string, qLst, cap = False ):
    """ Return True if `string` contains any member of `qLst` """
    for q in qLst:
        if cap and (q in string):
            return True
        elif (not cap) and (f"{q}".lower() in f"{string}".lower()):
            return True
    return False


def filter_series( series, stdFactor = 2.0 ):
    """ Filter outliers more than `stdFactor` standard deviations from the mean """
    if not len( series ):
        return list()
    mu = np.mean( series )
    sd = np.std( series )
    nuSeries = deque()
    thresh   = abs(sd*stdFactor)
    for datum in series:
        if abs( datum - mu ) <= thresh:
            nuSeries.append( datum )
    return list( nuSeries )


def as_json( dataObj, asLeaf = False ):
    """ Convert the dataclass into a struct that can be JSON serialized """
    def make_serializable( obj, depth = 0 ):
        nonlocal asLeaf
        if isinstance( obj, (deque, list,) ):
            rtnLst = deque()
            for item in obj:
                rtnLst.append( make_serializable( item, depth+1 ) )
            return list( rtnLst )
        elif isinstance( obj, np.ndarray ):
            return obj.tolist()
        elif isinstance( obj, dict ):
            rtnDct = dict()
            for k, v in obj.items():
                rtnDct[k] = make_serializable( v, depth+1 )
            return rtnDct
        else:
            return obj
    rtnObj = make_serializable( dataObj, 0 )
    return rtnObj


def crash_out():
    """ End the program with Brutal Finality """
    print( "\n\n" )
    os.system( 'kill %d' % os.getpid() ) 


def extract_pose_from_str( poseStr : str ):
    """ Get the homogeneous coordinates from the string and ignore everything else """
    nstLst = list()
    depth  = 0
    numStr = ""
    row    = list()

    def store_num():
        """ Add the number to the row """
        nonlocal row, numStr, poseStr
        if len( numStr ):
            try:
                row.append( float( numStr.strip() ) )
            except ValueError as e:
                print( f"BAD: {e}" )
                print( numStr  )
                print( poseStr )
                crash_out()
        numStr = ""

    def store_row():
        """ Add the row to the array """
        nonlocal nstLst, row
        if len( row ):
            nstLst.append( row )
        row = list()

    for char in poseStr:
        if char == '[':
            depth += 1
        elif char == ']':
            if depth == 2:
                store_num()
            depth -= 1
            if depth == 1:
                store_row()
        elif char == ' ':
            if depth == 2:
                store_num()
        elif char == '\n':
            pass
        elif depth == 2:
            numStr += char
        else:
            pass
            # print( f"`extract_pose_from_str()`, BAD STATE:\n{char}\n{poseStr}\n" )

    try:
        return np.array( nstLst )
    except Exception as e:
        traceback.print_exc()
        print( f"BAD: {e}" )
        crash_out()



########## HELPER CLASSES ##########################################################################

class SymbolHistory:
    """ Simple class for tracking Confusion and Other Symbol Problems """
    def __init__( self ):
        self.hist : Deque[GraspObj]            = deque()
        self.plns : Deque[dict[str,list[str]]] = deque()


    def ingest_frame( self, objList : list[GraspObj] ):
        """ Log symbols for eval later """
        self.hist.append( deep_copy_memory_list( objList ) )


    def ingest_plan( self, plan : dict[str,list[str]] ):
        """ Log plan for eval later """
        self.plns.append( deepcopy( plan ) )


    @staticmethod
    def action_2_move( action : dict[str,list[str]] ):
        """ Express the action as a move from one pose to another """
        if 'next' in action:
            seq = action['next']
        elif 'plan' in action:
            seq = action['plan'][:4]
        else:
            return None
        src = np.eye(4)
        dst = np.eye(4)
        for bhv in seq:
            if "Pick" in bhv[:10]:
                src = extract_pose_from_str( bhv )
            elif ("Place" in bhv[:10]) or ("Stack" in bhv[:10]):
                dst = extract_pose_from_str( bhv )
        return {"src" : src, "dst" : dst,}
    

    def last_frame_confusion( self ):
        """ Return a count of the unmoved blocks that have changed identities, Also return object count last step """
        # Init
        move = None
        if len( self.plns ):
            move = SymbolHistory.action_2_move( self.plns[-1] )
        currObjs : list[GraspObj] =  self.hist[-1] if (len(self.hist) >= 1) else list()
        prevObjs : list[GraspObj] =  self.hist[-2] if (len(self.hist) >= 2) else list()

        # Take action into account!
        if move is not None:
            for objPrv in prevObjs:
                if euclidean_distance_between_symbols( objPrv, move["src"] ) <= env_var("_ACCEPT_POSN_ERR"):
                    objPrv.pose = ObjPose( move["dst"] )
        
        # For each previous object, Search for a matching current object
        closest : dict[int,dict[str,GraspObj]] = dict() 
        for obj_j in prevObjs:
            dMin = 6e10
            oMin = None
            for obj_i in currObjs:
                d_ij = euclidean_distance_between_symbols( obj_i, obj_j )
                if d_ij <= env_var("_ACCEPT_POSN_ERR"):
                    if d_ij < dMin:
                        dMin = d_ij
                        oMin = obj_i
            if oMin is not None:
                closest[ id(oMin) ] = { "prev" : obj_j, "curr" : oMin } 
        
        # Evaluate Matches
        Ncurr  = len( currObjs )
        Nconf = 0
        for pair in closest.values():
            if pair["prev"].label != pair["curr"].label:
                Nconf += 1
        
        # Return confused and total
        return Nconf, Ncurr



########## SAVE: DATA PROCESSING ###################################################################
_SAVE_DATA = True
_LOAD_DATA = True 

if _SAVE_DATA:
    try:
        totRes = dict()

        for iii, paths in enumerate( datasets ):
            setNam = dataLabels[iii]
            suffix = "_" + setNam
            skip   = False
            totRes[ setNam ] = dict()

            for ii, test in enumerate( tests ):
                ##### Init ################################################################
                path     = paths[ii]
                fName    = fNames[ii]
                longTNam = longTestNames[ii]

                ##### Load ################################################################
                try:
                    pkls = [os.path.join( path, item ) for item in os.listdir( path ) if ".pkl" in f"{item}".lower()]
                except FileNotFoundError as e:
                    print( f"\n404, SKIP THIS TEST: {e}\n" )
                    skip = True
                    continue

                ########## ANALYSIS ####################################################################
                results = {
                    ### Steps ###
                    "Nstep"   : deque(),
                    "tStep"   : deque(),
                    "tEpisd"  : deque(),
                    "rSuccess": deque(),
                    ### 1. Object Search ###
                    "tSearch": deque(),
                    ### 2. Symbol Grounding ###
                    "tGround"  : deque(),
                    "rGround"  : deque(),
                    "rConfuse" : deque(),
                    "rFindFail": deque(),
                    ### 3. Planning ###
                    "tPlan"    : deque(),
                    "rPlan"    : deque(),
                    "rPlanFail": deque(),
                    ### 4. Acting ###
                    "tAct"    : deque(),
                    "rAct"    : deque(),
                    "rActFail": deque(),
                    ### 5. Resetting ###
                    "tReset": deque(),
                    "rReset": deque(),
                }

                ##### Read Data ###############################################

                for dPth in pkls:
                    print( f"About to open {dPth} ..." )
                    data = list()

                    try:
                        with open( dPth, 'rb' ) as f:
                            data = pickle.load( f )
                    except EOFError as e:
                        print( f"LOAD ERROR: {e}" )
                        continue

                    ##### Per-Episode Accounting ##################################
                
                    ### Steps ###
                    Nstep    = 0
                    tStepBgn = 0
                    tStepEnd = 0
                    tStepDqu = deque()

                    ### 1. Object Search ###
                    tSearchBgn = 0
                    tSearchEnd = 0
                    tSearchDqu = deque()

                    ### 2. Symbol Grounding ###
                    Nground    = 0
                    tGroundBgn = 0
                    tGroundEnd = 0
                    tGroundDqu = deque()
                    symbols_t  = list()
                    symHst     = SymbolHistory()
                    totFound   = 0
                    totConfuse = 0
                    NfailFind  = 0

                    ### 3. Planning ###
                    Nplan     = 0
                    tPlanBgn  = 0
                    tPlanEnd  = 0
                    tPlanDqu  = deque()
                    NfailPlan = 0
                    pNewStep  = False # Has a new step begun?
                    pTryPlan  = False # Did we try to plan this step?

                    ### 4. Acting ###
                    Naction    = 0
                    tActionBgn = 0
                    tActionEnd = 0
                    tActionDqu = deque()
                    NfailActn  = 0

                    ### 5. Resetting ###
                    Nreset    = 0
                    tResetBgn = 0
                    tResetEnd = 0
                    tResetDqu = deque()

                    ##### Per-Message Accounting #####
                    # ASSUMPTION: "BGN: ..." / "END: ..." MESSAGES ALWAYS APPEAR IN THE CORRECT ORDER! 
                    for datum in data:
                        dtmMsg = datum['msg']
                        dtmT   = datum['t']
                        dtmDat = datum['data']

                        ##### Phase 1: Perception #############################
                        if "BGN: Phase 1" in dtmMsg:
                            # ASSUMPTION: PHASE 1 MESSAGE SENT ONLY ONCE PER STEP, See `p1pp2`
                            Nstep += 1
                            if tStepBgn > 0:
                                tStepEnd = dtmT
                                tStepDqu.append( tStepEnd - tStepBgn )
                            tStepBgn   = dtmT
                            tSearchBgn = dtmT
                            
                            # WARNING: THIS SMELLS
                            if pNewStep and (not pTryPlan):
                                NfailFind += 1
                            pNewStep = True
                            pTryPlan = False

                        if "END: Phase 1" in dtmMsg:
                            tSearchEnd = dtmT
                            tSearchDqu.append( tSearchEnd - tSearchBgn )

                        
                        ##### Phase 2: Grounding ##############################
                        if "BGN: Phase 2" in dtmMsg:
                            Nground   += 1
                            tGroundBgn = dtmT

                        if "Conditions Grounded" in dtmMsg:
                            pass
                            # symSet   = set([])
                            # for item in dtmDat:
                            #     if item[0] == "GraspObj":
                            #         symSet.add( item[1] )

                            # if len( symSet ) < 3:
                            #     # print( symSet ) # It's all or nothing, it seems
                            #     # print( dtmDat )
                            #     NfailFind += 1

                        if "END: Phase 2" in dtmMsg:
                            tGroundEnd = dtmT
                            tGroundDqu.append( tGroundEnd - tGroundBgn )
                            symbols_t = dtmDat[:]
                            symHst.ingest_frame( dtmDat[:] )
                            if len( symbols_t ):
                                Nconf, Nfram = symHst.last_frame_confusion()
                                totFound   += Nfram
                                totConfuse += Nconf


                        ##### Phase 3: Planning ###############################
                        if "BGN: Phase 3" in dtmMsg:
                            Nplan += 1
                            tPlanBgn = dtmT
                            if pNewStep:
                                pNewStep = False
                                pTryPlan = True

                        if "Planning Failure" in dtmMsg:
                            NfailPlan += 1

                        if "END: Phase 3" in dtmMsg:
                            tPlanEnd = dtmT
                            tPlanDqu.append( tPlanEnd - tPlanBgn )
                            if len( dtmDat ):
                                symHst.ingest_plan( dtmDat )
                            


                        ##### Phase 4: Execution ##############################
                        if "BGN: Phase 4" in dtmMsg:
                            Naction   += 1
                            tActionBgn = dtmT

                        if ("BT END" in dtmMsg) and ("fail" in f"{dtmMsg}".lower()):
                            NfailActn += 1

                        if "END: Phase 4" in dtmMsg:
                            tActionEnd = dtmT
                            tActionDqu.append( tActionEnd - tActionBgn )


                        ##### Phase 5: Reset ##################################
                        if "BGN: Phase 5" in dtmMsg:
                            Nreset   += 1
                            tResetBgn = dtmT

                        if "END: Phase 5" in dtmMsg:
                            tResetEnd = dtmT
                            tResetDqu.append( tResetEnd - tResetBgn )
                            
                    
                    ##### Per-Episode Accounting ##################################
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
                        

                    ### Steps ###
                    results["Nstep"].append( Nstep )
                    results["tStep"].extend( tStepDqu )
                    results["tEpisd"].append( data[-1]['t'] - data[0]['t'] )
                    results["rSuccess"].append( resEp )
                    ### 1. Object Search ###
                    results["tSearch"].extend( tSearchDqu )
                    ### 2. Symbol Grounding ###
                    results["tGround"].extend( tGroundDqu )
                    results["rGround"].append( Nground / Nstep )
                    results["rConfuse"].append( totConfuse / totFound )
                    results["rFindFail"].append( NfailFind / Nground )
                    ### 3. Planning ###
                    results["tPlan"].extend( tPlanDqu )
                    results["rPlan"].append( Nplan / Nstep )
                    results["rPlanFail"].append( NfailPlan / Nplan )
                    ### 4. Acting ###
                    results["tAct"].extend( tActionDqu )
                    results["rAct"].append( Naction / Nstep )
                    results["rActFail"].append( NfailActn / Naction )
                    ### 5. Resetting ###
                    results["tReset"].extend( tResetDqu )
                    results["rReset"].append( Nreset / Nstep )

                # pprint( results )
                totRes[ setNam ][ test ] = deepcopy( results )

    except KeyboardInterrupt:
        crash_out()

    ##### Store Results ###################################################
    with open( _JSON_PATH, 'w' ) as f:
        json.dump( as_json( totRes ), f, indent = 2 )
                        



########## LOAD: GRAPHICS ##########################################################################

if _LOAD_DATA:
    try:
        with open( _JSON_PATH, 'r' ) as f:
            totRes = json.load( f )

        for iii, paths in enumerate( datasets ):
            setNam = dataLabels[iii]
            suffix = "_" + setNam

            for ii, test in enumerate( tests ):
                ##### Init ################################################################
                path     = paths[ii]
                fName    = fNames[ii]
                longTNam = longTestNames[ii]
                results  = totRes[ setNam ][ test ]

                _FILTER_FACTOR = 2.5    
                results['tSearch'] = filter_series( results['tSearch'], _FILTER_FACTOR )
                results['tEpisd']  = filter_series( results['tEpisd'], _FILTER_FACTOR )


                ##### Sanity Checks ##################
                print( f"Scenario normally takes {np.mean(results['Nstep'])} steps!, Success Rate: {np.mean(results['rSuccess']):.3f}" )
                print( f"Phase 1: 1.000 --> Phase 2: {np.mean(results['rGround']):.3f} --> Phase 3: {np.mean(results['rPlan']):.3f} --> Phase 4: {np.mean(results['rAct']):.3f} --> Phase 5: {np.mean(results['rReset']):.3f}" )
                # print( f'{np.mean(results["rFindFail"]):.3f} + {np.mean(results["rPlan"]):.3f} + {(1.0-np.mean(results["rSuccess"]))/np.mean(results["Nstep"])} == 1.000: Error: {1.0 - np.mean(results["rFindFail"]) - np.mean(results["rPlan"]) - (1.0-np.mean(results["rSuccess"]))/np.mean(results["Nstep"])}' )
                
                # fullCount = deque()
                # for i in range( len( results["rFindFail"] ) ):
                #     fullCount.append(
                #         results["rFindFail"][i] + results["rPlan"][i]
                #     )
                skipCount = deque()
                for i in range( len( results["Nstep"] ) ):
                    skip = 0.0
                    # skip = -1 / results["Nstep"][i]
                    # skip = -1 
                    if results["rSuccess"][i]:
                        skip = 1 / results["Nstep"][i]
                    skipCount.append( skip )
                    
                
                # print( f'{np.mean(results["rFindFail"]):.3f} + {np.mean(results["rPlan"]):.3f} == 1.000: Error: {1.0 - np.mean(results["rFindFail"]) - np.mean(results["rPlan"])}' )
                print( f'{np.mean(results["rFindFail"]):.3f} + {np.mean(results["rPlan"]):.3f} + {np.mean(skipCount)} == 1.000: Error: {1.0 - np.mean(results["rFindFail"]) - np.mean(results["rPlan"]) - np.mean(skipCount)}' )

                ##### Paper Plots ####################
                make_histo( results['Nstep'], f"{longTNam}, {suffix[1:]}\nMakespan Distribution [Steps]", f"{fName}_Histo-Steps{suffix}{plotExt}" )
                make_histo( results['tEpisd'], f"{longTNam}, {suffix[1:]}\nMakespan Distribution [Time]", f"{fName}_Histo-Time{suffix}{plotExt}" )
                make_histo( results['tSearch'], f"{longTNam}, {suffix[1:]}\nObject Search Time Distribution", f"{fName}_Histo-Search{suffix}{plotExt}", 
                            xLabel = 'Time [s]', forceYlim = False )
                make_histo( results['tAct'], f"{longTNam}, {suffix[1:]}\nAction Execution Time Distribution", f"{fName}_Histo-ActionPass{suffix}{plotExt}", 
                            xLabel = 'Time [s]', forceYlim = False )
    except KeyboardInterrupt:
        crash_out()



########## EXIT ####################################################################################
crash_out()