########## INIT ####################################################################################
import pickle, os
from collections import deque
from pprint import pprint


import numpy as np

from aspire.env_config import env_var
from aspire.symbols import GraspObj, euclidean_distance_between_symbols
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import set_experiment_env
from draw_beliefs import ( set_render_env, render_memory_list, scan_geo, vispy_geo_list_window, cpcd_geo )

##### Environment && Constants ############################################
set_blocks_env()
set_experiment_env()
set_render_env()


########## SETUP ###################################################################################
_DATA_DRIVE = "DATA_TANK"
_PLOT_DIR   = "/media/james/FILEPILE/EROM/data/plots/"

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

paths = [ f"/media/james/{_DATA_DRIVE}/2025-08_{test}" for test in tests ]

fNames = [ f"{_PLOT_DIR}{test}" for test in tests  ]

plotExt = ".pdf"


########## HELPER FUNCTIONS ########################################################################
import matplotlib.pyplot as plt


_TITLE_FONT_SIZE = 13


def make_histo( series, plotTitle, fName, showPlot = False, xLabel = 'Makespan', yLabel = 'Occurrences' ):
    """ Create Histogram """
    plt.clf()
    print( f"\n{plotTitle}" )
    print( f"Mean: ___ {np.mean(series)}" )
    print( f"Median: _ {np.median(series)}" )
    print( f"Std.Dev.: {np.std(series)}" )

    plt.hist( series )
    plt.title( plotTitle, fontsize = _TITLE_FONT_SIZE ) # Set the title && font size
    plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    plt.savefig( fName )
    if showPlot:
        plt.show() 


def make_multi_histo( multiSeries, seriesNames, plotTitle, fName, showPlot = False, xLabel = 'Makespan', yLabel = 'Occurrences' ):
    """ Create Histogram """
    plt.clf()
    print( f"\n### {plotTitle} ###" )
    for i, series in enumerate( multiSeries ):
        print( f"\t{seriesNames[i]}" )
        print( f"\tMean: ___ {np.mean(series)}" )
        print( f"\tMedian: _ {np.median(series)}" )
        print( f"\tStd.Dev.: {np.std(series)}" )
    plt.hist( multiSeries, label = seriesNames )
    plt.title( plotTitle, fontsize = _TITLE_FONT_SIZE ) # Set the title && font size
    plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    plt.legend( loc = 'upper right' )
    plt.savefig( fName )
    if showPlot:
        plt.show() 


def make_scatter( X, Y, plotTitle, fName, showPlot = False ):
    """ Create Histogram """
    plt.clf()
    print()

    plt.scatter( X, Y )
    plt.title( plotTitle ) # Set the title
    plt.xlabel('Step')  # Setting the x-axis label
    plt.ylabel('Time') # Setting the y-axis label
    plt.savefig( fName )
    if showPlot:
        plt.show() 


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

for ii, test in enumerate( tests ):
    ##### Init ################################################################
    path     = paths[ii]
    fName    = fNames[ii]
    longTNam = longTestNames[ii]

    ##### Load ################################################################
    pkls = [os.path.join( path, item ) for item in os.listdir( path ) if ".pkl" in f"{item}".lower()]


    ########## ANALYSIS ################################################################################

    ##### Success Rate && Makespan ############################################
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
            'N'      : deque(),
        }, 
    }

    for dPth in pkls:
        print( f"About to open {dPth} ..." )
        data = list()
        

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

        # Segmentation Performance
        obsTimes = deque()
        obsBgn   = 0
        obsEnd   = 0
        obsRun   = False

        # Confusion Tracking
        symHist = deque()
        symDlta = deque()
        Ndelta  = 0
        Ncnfus  = 0
        Ntotal  = 0

        for i, datum in enumerate( data ):

            if datum['msg'] == "Observation BEGIN":
                if not obsRun:
                    obsBgn = datum['t']
                obsRun = True
            elif p_str_has_any( datum['msg'], ["BT BEGIN", "Planning Failure"], cap = False ):
                if obsRun:
                    obsEnd   = datum['t']
                    duration = obsEnd - obsBgn
                    obsTimes.append( duration )
                    result['oStp']['s'].append( Nstp     )
                    result['oStp']['t'].append( duration )
                obsRun = False
            
            if p_str_has_any( datum['msg'], ["BT END", "Planning Failure"], cap = False ):
                result['sDel']['s'].append( Nstp   ) 
                result['sDel']['c'].append( Ndelta )
                Nstp  += 1
                Ndelta = 0 # Reset confusions for the next step

                if "Planning Failure" in datum['msg']:
                    NfailPlan += 1

                if ("BT END" in datum['msg']) and ("fail" in f"{datum['msg']}".lower()):
                    NfailActn += 1
            

            if ('symbol' in datum['msg']):
                if len( datum['data'] ):
                    # print( f"{Ncnfus}:{Ntotal}\n{datum['data']}" )
                    symbols : list[GraspObj] = datum['data']
                    if len( symHist ) and len( symHist[-1] ):
                        symLast : list[GraspObj] = symHist[-1]
                        Nmatch = 0
                        mtcSet = set([])
                        for sym_i in symbols:
                            Ntotal += 1
                            dMin_i = 6e10
                            mtch_i = None
                            for sym_j in symLast:
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

        # print( f"\nEnd of Episode Confusion: {Ncnfus}:{Ntotal} = {Ncnfus / Ntotal}\n" )

        result['sRun'].append( Nstp )
        result['tRun'].append( data[-1]['t'] - data[0]['t'] )
        result['rCon'].append( Ncnfus / Ntotal )
        result['rFal']['action'].append( NfailActn/Nstp )
        result['rFal']['plan'  ].append( NfailPlan/Nstp )
        result['rFal']['N'     ].append( Nstp           )
        result['tObs'].extend( obsTimes )
    
    _FILTER_FACTOR = 2.5    
    # result['sRun'] = filter_series( result['sRun'], _FILTER_FACTOR )
    # result['tRun'] = filter_series( result['tRun'], _FILTER_FACTOR )    
    result['tObs'] = filter_series( result['tObs'], _FILTER_FACTOR )

    make_histo( result['sRun'], f"{longTNam},\nMakespan Distribution [Steps]", f"{fName}_Histo-Steps{plotExt}" )
    make_histo( result['tRun'], f"{longTNam},\nMakespan Distribution [Time]", f"{fName}_Histo-Time{plotExt}" )
    make_histo( result['tObs'], f"{longTNam},\nObject Search Time Distribution", f"{fName}_Histo-Search{plotExt}", xLabel = 'Time [s]' )
    make_histo( result['rCon'], f"{longTNam},\nObject Confusion Distribution", f"{fName}_Histo-Confusion{plotExt}", xLabel = 'Confusion Rate' )
    make_scatter( result['oStp']['s'], result['oStp']['t'], 
                  f"{longTNam},\nObject Search Time at Each Step", f"{fName}_Scatter-Search{plotExt}" )
    make_scatter( result['sDel']['s'], result['sDel']['c'], 
                  f"{longTNam},\nNumber of Objects Confused at Each Step", f"{fName}_Scatter-Confusion{plotExt}" )
    make_multi_histo( 
        [ result['rFal']['action'], result['rFal']['plan'], ], 
        [ "Action Failure Rate", "Planning Failure Rate", ], 
        f"{longTNam},\nDistribution of Action and Planning Failure Rates, Per Episode", 
        f"{fName}_Histo-Failure{plotExt}", 
        xLabel = 'Failure Rates', 
        yLabel = 'Occurrences' 
    )



########## EXIT ####################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 