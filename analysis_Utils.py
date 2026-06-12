from __future__ import annotations
import os

import matplotlib.pyplot as plt
import numpy as np

from aspire.BlocksTask import set_blocks_env
from aspire.env_config import env_var
from TaskPlanner import set_experiment_env
from draw_beliefs import set_render_env

_TITLE_FONT_SIZE =  13
_TIGHT_MARGIN    =   0.05
_DEFAULT_DIV     = 100 #80 #100 #200

##### Environment && Constants ############################################
set_blocks_env()
set_experiment_env()
set_render_env()

########## HELPER FUNCTIONS ########################################################################

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


def dex_key( x, offset = -1 ):
    dex = f"{x}".split('_')[ offset ].replace( ".pkl", "" )
    if len( dex ) >= 2:
        return dex
    elif len( dex ) < 2:
        return '0'*(2-len( dex )) + dex
    else:
        raise ValueError( "`dex_key`: This should NOT have happened!" )


def play_tone( duration_s = 5, freq_Hz = 650 ):
    """ Play a notification tone """
    os.system( f'play -nq -t alsa synth {duration_s} sine {freq_Hz}' )


def crash_out( notify = True ):
    """ End the program with Brutal Finality """
    if notify:
        play_tone()
    print( "\n\n" )
    os.system( 'kill %d' % os.getpid() ) 


def xy_plot_filled_under( X, Y, plotTitle = None, fName = "output.pdf", xLabel = None, yLabel = None, 
                          titleFontSize_pt = _TITLE_FONT_SIZE ):
    """ Creat cumulative curve """
    # Plot line
    plt.plot( X, Y )

    # Shade the area under the curve
    plt.fill_between( X, Y, 0, color = 'skyblue', alpha = 0.5 )

    if plotTitle is not None:
        plt.title( plotTitle, fontsize = titleFontSize_pt ) # Set the title && font size
    if xLabel is not None:
        plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    if yLabel is not None:
        plt.ylabel( yLabel ) # ---------------- Setting the y-axis label

    plt.show()


def make_histo( series, plotTitle, xLabel = 'Makespan', yLabel = 'Occurrences', savefig = True ):
    """ Create Histogram """
    if savefig:
        plt.clf()
    plt.margins( _TIGHT_MARGIN )
    print( f"\n{plotTitle}" )
    print( f"Mean: ___ {np.mean(series)}" )
    print( f"Median: _ {np.median(series)}" )
    print( f"Std.Dev.: {np.std(series)}" )
    plt.hist( series, _DEFAULT_DIV )
    plt.title( plotTitle, fontsize = _TITLE_FONT_SIZE ) # Set the title && font size
    plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    plt.tight_layout()
    plt.show()



########## LOCATION ################################################################################
_DATA_DRIVE = "STARGAZER/DATA_TANK"

class Loc:
    """ Static Container Class """    

    _MISC_DIR  = "/media/james/STARGAZER/DATA_TANK/misc_data/" 
    _PLOT_DIR  = "/media/james/FILEPILE/EROM/data/plots/"
    _GC_CYCLE  = False 
    _F_EXTRACT = f"{_PLOT_DIR}outData.pkl"
    _T_EXTRACT = f"{_PLOT_DIR}outText.json"

    _MIN_STATE_SIZE_BYTES = 500.0

    _TITLE_FONT_SIZE = 13
    _TIGHT_MARGIN    =  0.05

    _DEFAULT_DIV = 100 #80 #100 #200

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

    eBlcNam = {
        "RGB": ['redBlock', 'grnBlock', 'bluBlock', env_var("_NULL_NAME"),], 
        "RBW": ['redBlock', 'blkBlock', 'whtBlock', env_var("_NULL_NAME"),],
    }

    plotExt = ".pdf"