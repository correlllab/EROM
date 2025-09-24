########## INIT ####################################################################################
import pickle, os, math, json
from collections import deque
from copy import deepcopy
from pprint import pprint

import numpy as np

from aspire.env_config import env_var
from aspire.symbols import GraspObj, euclidean_distance_between_symbols
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import set_experiment_env
from draw_beliefs import set_render_env

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

fNames = [ f"{_PLOT_DIR}{test}" for test in tests  ]

plotExt = ".pdf"


########## HELPER FUNCTIONS ########################################################################
import matplotlib.pyplot as plt


_TITLE_FONT_SIZE = 13
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


def make_multi_histo( multiSeries, seriesNames, plotTitle = None, fName = "output.pdf", xLabel = 'Makespan', yLabel = 'Occurrences', 
                      forceYlim = True, savefig = True ):
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
        plt.title( plotTitle, fontsize = _TITLE_FONT_SIZE ) # Set the title && font size
    plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    if savefig:
        plt.legend( loc = 'upper right' )
    if forceYlim:
        plt.ylim( (0, _N_TRIALS,) )
    plt.tight_layout()
    if savefig:
        plt.savefig( fName )
        return plt.gca()



# def make_scatter( X, Y, plotTitle, fName, showPlot = False ):
#     """ Create Histogram """
#     plt.clf()
#     print()

#     plt.scatter( X, Y )
#     plt.title( plotTitle ) # Set the title
#     plt.xlabel('Step')  # Setting the x-axis label
#     plt.ylabel('Time') # Setting the y-axis label
#     plt.savefig( fName )
#     if showPlot:
#         plt.show() 


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

_D_THRESH_M = env_var("_BLOCK_SCALE")*0.75


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
    pprint( rtnObj )
    return rtnObj



########## SAVE: DATA PROCESSING ###################################################################
_SAVE_DATA = False
_LOAD_DATA = True 

if _SAVE_DATA:
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

            ########## ANALYSIS ################################################################################

            ##### Success Rate && Makespan Tracking ###############################
            N   = 0
            S   = 0
            F   = 0
            MTS = 0.0
            MTF = 0.0

            result = {
                'Ntrial' : 0,
                'outcome': deque(),
                'tRun'   : deque(),
                'sRun'   : deque(),
                'tObs'   : deque(),
                'oStp'   : {
                    "s": deque(),
                    "t": deque(),
                },
                'sDel' : {
                    "s": deque(),
                    "c": deque(),
                },
                'rCon': deque(),
                'rFal': {
                    'action' : deque(),
                    'plan'   : deque(),
                    'find'   : deque(),
                    'N'      : deque(),
                },
            }

            total = {
                'Ntrial' : 0,
                'outcome': deque(),
                'tRun'   : deque(),
                'sRun'   : deque(),
            }

            ##### Per-Episode Accounting ##########################################

            for dPth in pkls:
                print( f"About to open {dPth} ..." )
                data = list()
                N   += 1

                try:
                    with open( dPth, 'rb' ) as f:
                        data = pickle.load( f )
                except EOFError as e:
                    print( f"LOAD ERROR: {e}" )
                    continue

                # Failure Tracking
                Nstp      = 0
                NfailActn = 0
                NfailPlan = 0
                NfailFind = 0

                # Segmentation Performance
                obsTimes = deque()
                obsBgn   = 0
                obsEnd   = 0

                # Confusion Tracking
                symHist = deque()
                symDlta = deque()
                Ndelta  = 0
                Ncnfus  = 0
                Ntotal  = 0
                symbols : list[GraspObj] = list()

                txtPath = dPth.split('/')[-1].split('.')[0] + ".txt"
                outFile = open( f"data/txt_log/{txtPath}", 'w' )
                timeBgn = data[0]['t']

                ##### Per-Step Accounting ################

                for datum in data:
                    # print( ".", end="", flush=True )
                    dtmMsg = datum['msg']
                    dtmT   = datum['t']

                    outFile.write( f"{(datum['t']-timeBgn):08.2f} : {dtmMsg}\n" )

                    ##### Failures #####

                    if "Planning Failure" in dtmMsg:
                        NfailPlan += 1

                    if ("BT END" in dtmMsg) and ("fail" in f"{dtmMsg}".lower()):
                        NfailActn += 1

                    if p_str_has_any( dtmMsg, ["BT END", "Planning Failure"], cap = False ):
                        Nstp += 1

                    ##### Symbol Grounding #####

                    if p_str_has_any( dtmMsg, ["symbols", "Post-Cheat"], cap = False ): # NOTE: Cheat ALWAYS applied in some form, even if fixes aren't applied
                        symbols = datum['data']
                        
                    ##### Phase 1: Perception #####

                    if "BGN: Phase 1" in dtmMsg:
                        obsBgn = dtmT
                    elif "END: Phase 1" in dtmMsg:
                        obsEnd = dtmT
                        duration = obsEnd - obsBgn
                        obsTimes.append( duration )
                        result['oStp']['s'].append( Nstp     )
                        result['oStp']['t'].append( duration )

                    ##### Phase 2: Conditions #####

                    elif "BGN: Phase 2" in dtmMsg:
                        Nlabel  = 0
                        Ntotal += len( symbols )
                        found   = True
                        if len( symbols ):
                            Nlabel = len( set([item.label for item in symbols]) )
                        if Nlabel < 3:
                            NfailFind += 1
                            found = False

                        if len( symbols ):
                            if len( symHist ) and len( symHist[-1] ):
                                symLast : list[GraspObj] = symHist[-1]
                                Nmatch = 0
                                mtcSet = set([])
                                if found:
                                    for sym_i in symLast:
                                        dMin_i = 6e10
                                        mtch_i = None
                                        for sym_j in symbols:
                                            d_ij = euclidean_distance_between_symbols( sym_i, sym_j )
                                            if ((d_ij < dMin_i) and (d_ij <= _D_THRESH_M)) and (id(sym_j) not in mtcSet):
                                                dMin_i = d_ij
                                                mtch_i = sym_j
                                        if mtch_i is not None:
                                            mtcSet.add( id(mtch_i) )
                                            Nmatch += 1
                                            if sym_i.label != mtch_i.label:
                                                Ndelta += 1
                                                Ncnfus += 1
                            symHist.append( symbols[:] )
                            result['sDel']['s'].append( Nstp   ) 
                            result['sDel']['c'].append( Ndelta )
                        Ndelta = 0 # Reset confusions for the next step
                        
                        

                Nstp = Nstp if (Nstp > 0) else math.nan
                result['sRun'].append( Nstp )
                result['tRun'].append( data[-1]['t'] - data[0]['t'] )
                result['rCon'].append( (Ncnfus / Ntotal) if (Ntotal > 0) else math.nan )
                result['rFal']['action'].append( NfailActn/Nstp )
                result['rFal']['plan'  ].append( NfailPlan/Nstp )
                result['rFal']['find'  ].append( NfailFind/Nstp )
                result['rFal']['N'     ].append( Nstp           )
                result['tObs'].extend( obsTimes )

                tRun = data[-1]['t'] - data[0]['t']
                end  =  False
                total['Ntrial'] += 1
                total['tRun'  ].append( tRun )
                for i in range(1,11):
                    try:
                        msg = data[-i]['msg']
                    except IndexError as e:
                        print(e)
                        break
                    # print( msg )
                    if ("Status.FAILURE" in msg) and ("BT END" not in msg) and ("Behavior" not in msg):
                        F += 1
                        # print( f"FAILURE: {msg}" )
                        print( f"FAILURE" )
                        MTF += tRun
                        end = True
                        total['outcome'].append( 0 )
                        break
                    elif ("Status.SUCCESS" in msg) and ("BT END" not in msg) and ("Behavior" not in msg):
                        S += 1
                        # print( f"SUCCESS: {msg}" )
                        print( f"SUCCESS" )
                        MTS += tRun
                        end = True
                        total['outcome'].append( 1 )
                        break
                if not end:
                    F += 1
                    # print( f"FAILURE: {msg}" )
                    print( f"FAILURE" )
                    MTF += tRun
                print()

                outFile.close()

            if S > 0:
                MTS /= S
            if F > 0:
                MTF /= F
            print( f"\n{N} episodes, Success Rate: {S*1.0/N}, Failure Rate: {F*1.0/N}, Sanity Check == 0.0: {1.0-S*1.0/N-F*1.0/N}" )
            print( f"Mean Time to Success: {[int(item) for item in divmod( MTS, 60.0 )]}, Mean Time to Failure: {[int(item) for item in divmod( MTF, 60.0 )]}" )
            print( "\n\n" )

            totRes[ setNam ][ test ] = deepcopy( result )

    with open( _JSON_PATH, 'w' ) as f:
        json.dump( as_json( totRes ), f, indent = 2 )



########## LOAD: GRAPHICS ##########################################################################

if _LOAD_DATA:

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
            result   = totRes[ setNam ][ test ]

            
            _FILTER_FACTOR = 2.5    
            result['tObs'] = filter_series( result['tObs'], _FILTER_FACTOR )
            result['tRun'] = filter_series( result['tRun'], _FILTER_FACTOR )

            make_histo( result['sRun'], f"{longTNam}, {suffix[1:]}\nMakespan Distribution [Steps]", f"{fName}_Histo-Steps{suffix}{plotExt}" )
            make_histo( result['tRun'], f"{longTNam}, {suffix[1:]}\nMakespan Distribution [Time]", f"{fName}_Histo-Time{suffix}{plotExt}" )
            make_histo( result['tObs'], f"{longTNam}, {suffix[1:]}\nObject Search Time Distribution", f"{fName}_Histo-Search{suffix}{plotExt}", 
                        xLabel = 'Time [s]', forceYlim = False )
            make_histo( result['rCon'], f"{longTNam}, {suffix[1:]}\nObject Confusion Distribution", f"{fName}_Histo-Confusion{suffix}{plotExt}", 
                        xLabel = 'Confusion Rate' )
            make_multi_histo( 
                [ result['rFal']['action'], result['rFal']['plan'], result['rFal']['find'], ], 
                [ "Action Failure Rate", "Planning Failure Rate", "Search Failure Rate", ], 
                f"{longTNam}, {suffix[1:]}\nDistribution of Action and Planning Failure Rates, Per Episode", 
                f"{fName}_Histo-Failure{suffix}{plotExt}", 
                xLabel = 'Failure Rates', 
                yLabel = 'Occurrences' 
            )

        labels = deque()
        for test in tests:
            labels.append( test )
        fName  = f"Total{suffix}"

        def get_series( lblLst : list[str], key : str ):
            series = deque()
            for lbl in lblLst:
                series.append( totRes[ setNam ][ lbl ][key] )
            return list( series )


        make_multi_histo( 
            get_series( labels, 'sRun' ), 
            labels, 
            f"Makespan Distribution [Steps], {suffix[1:]}", 
            f"{_PLOT_DIR}/{fName}_Histo-MS-Steps{suffix}{plotExt}", 
            xLabel = 'Steps', 
            yLabel = 'Occurrences' 
        )

        make_multi_histo( 
            get_series( labels, 'tRun' ), 
            labels, 
            f"Makespan Distribution [Time], {suffix[1:]}", 
            f"{_PLOT_DIR}/{fName}_Histo-MS-Time{suffix}{plotExt}", 
            xLabel = 'Seconds', 
            yLabel = 'Occurrences' 
        )

        make_multi_histo( 
            get_series( labels, 'tObs' ), 
            labels, 
            f"Object Search Time Distribution, {suffix[1:]}", 
            f"{_PLOT_DIR}/{fName}_Histo-Search{suffix}{plotExt}", 
            xLabel = 'Seconds', 
            yLabel = 'Occurrences' 
        )

        make_multi_histo( 
            get_series( labels, 'rCon' ), 
            labels, 
            f"Object Confusion Distribution, {suffix[1:]}", 
            f"{_PLOT_DIR}/{fName}_Histo-Confusion{suffix}{plotExt}", 
            xLabel = 'Confusion Rate', 
            yLabel = 'Occurrences' 
        )

    # pprint( totRes )


    ##### Big Subplots ########################################################
    for dataName, dataset in totRes.items():
        plt.clf()
        for i in range(4):
            plt.subplot(2, 2, i+1)
            result = dataset[ tests[i] ] 
            make_multi_histo( 
                [ result['rFal']['action'], result['rFal']['plan'], result['rFal']['find'], ], 
                [ "Action Failure Rate", "Planning Failure Rate", "Search Failure Rate", ], 
                # f"{longTNam}, {suffix[1:]}\nDistribution of Action and Planning Failure Rates, Per Episode", 
                # f"{fName}_Histo-Failure{suffix}{plotExt}", 
                xLabel = 'Failure Rates', 
                yLabel = 'Occurrences',
                savefig = False
            )
            if i == 1:
                plt.legend( loc='upper right')
        plt.subplots_adjust(top=0.90) # Adjust this value as needed
        plt.suptitle( f"Failure Rates for {tests}" )
        plt.savefig( f"{_PLOT_DIR}/{dataName}_Failure-Rates{plotExt}" )



########## EXIT ####################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 