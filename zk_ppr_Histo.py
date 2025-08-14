########## INIT ####################################################################################
import pickle, os
from collections import deque
from pprint import pprint


import numpy as np

from aspire.symbols import GraspObj
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

paths = [ f"/media/james/{_DATA_DRIVE}/2025-08_{test}" for test in tests ]

plotTitles = [
    "Distribution with Known Class & Known Pose", 
    "Distribution with Sensed Class & Known Pose", 
    "Distribution with Known Class & Sensed Pose", 
    "Distribution with Sensed Class & Sensed Pose", 
]

fNames = [ f"{_PLOT_DIR}{test}_Histo-" for test in tests  ]

plotExt = ".pdf"


########## HELPER FUNCTIONS ########################################################################
import matplotlib.pyplot as plt


def make_histo( series, plotTitle, fName, showPlot = False ):
    """ Create Histogram """
    plt.clf()
    print()
    print( f"Mean: ___ {np.mean(series)}" )
    print( f"Median: _ {np.median(series)}" )
    print( f"Std.Dev.: {np.std(series)}" )

    plt.hist( series )
    plt.title( plotTitle ) # Set the title
    plt.xlabel('Makespan')  # Setting the x-axis label
    plt.ylabel('Occurrences') # Setting the y-axis label
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


for ii, test in enumerate( tests ):
    path      = paths[ii]
    plotTitle = plotTitles[ii]
    fName     = fNames[ii]

    ##### Load ################################################################
    pkls = [os.path.join( path, item ) for item in os.listdir( path ) if ".pkl" in f"{item}".lower()]
    # for pkl in pkls:
    #     print( pkl )


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
    }

    for dPth in pkls:
        print( f"About to open {dPth} ..." )
        data = list()
        Nstp = 0

        try:
            with open( dPth, 'rb' ) as f:
                data = pickle.load( f )
        except EOFError as e:
            print( f"LOAD ERROR: {e}" )
            continue

        obsTimes = deque()
        obsBgn   = 0
        obsEnd   = 0
        obsRun   = False

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
                    result['oStp']['s'].append( Nstp )
                    result['oStp']['t'].append( duration )
                obsRun = False
            
            if p_str_has_any( datum['msg'], ["BT END", "Planning Failure"], cap = False ):
                Nstp += 1


        result['sRun'].append( Nstp )
        result['tRun'].append( data[-1]['t'] - data[0]['t'] )
        result['tObs'].extend( obsTimes )
    
    _FILTER_FACTOR = 2.5    
    # result['sRun'] = filter_series( result['sRun'], _FILTER_FACTOR )
    # result['tRun'] = filter_series( result['tRun'], _FILTER_FACTOR )    
    result['tObs'] = filter_series( result['tObs'], _FILTER_FACTOR )
    
    make_histo( result['sRun'], f"{plotTitle}, Makespan Steps", f"{fName}Steps{plotExt}" )
    make_histo( result['tRun'], f"{plotTitle}, Makespan Time", f"{fName}Time{plotExt}" )
    make_histo( result['tObs'], f"{plotTitle}, Object Search Time", f"{fName}Search{plotExt}" )
    make_scatter( result['oStp']['s'], result['oStp']['t'], f"{plotTitle}, Object Search Trend", f"{fName}Trend{plotExt}" )



########## EXIT ####################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 