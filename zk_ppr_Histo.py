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
    "Makespan Distribution with Known Class & Known Pose", 
    "Makespan Distribution with Sensed Class & Known Pose", 
    "Makespan Distribution with Known Class & Sensed Pose", 
    "Makespan Distribution with Sensed Class & Sensed Pose", 
]

fNames = [ f"{_PLOT_DIR}{test}_Histo-" for test in tests  ]

plotExt = ".pdf"


########## PLOTS ###################################################################################
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

        for i, datum in enumerate( data ):
            if datum['msg'] == 'memory':
                Nstp += 1
                # print( f"\t{i}\t{datum['msg']}" )

        result['sRun'].append( Nstp )
        result['tRun'].append( data[-1]['t'] - data[0]['t'] )

    make_histo( result['sRun'], f"{plotTitle}, Steps", f"{fName}Steps{plotExt}" )
    make_histo( result['tRun'], f"{plotTitle}, Time", f"{fName}Time{plotExt}" )



########## EXIT ####################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 