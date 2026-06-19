from __future__ import annotations
import os, json
from collections import deque
from random import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import poisson, lognorm

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


def chop_sigmas( data : list[int], sigmas = 3.0, useMean : bool = True ) -> list[int]:
    """ Return a version of `binPopLst` without the most distant outliers """
    rtnL = deque()
    if useMean:
        mean = np.mean( data )
    else:
        mean = np.median( data )
    stdv = np.std( data )
    lo   = max( mean - sigmas * stdv, 0.0 )
    hi   = mean + sigmas * stdv
    for val in data:
        if lo <  val <= hi:
            rtnL.append( val )
    return list( rtnL )


def fit_lognorm_to_data( data : list[float], sigmaChop : float = 3.0, useMean : bool = True ) -> tuple[float,float,float]:
    shape, loc, scale = lognorm.fit( chop_sigmas( data, sigmaChop, useMean = useMean ), floc = 0 )
    print( f"Log-Normal Fit - Shape: {shape}, Location: {loc}, Scale: {scale}" )
    return shape, loc, scale



########## PROBABILITY CLASSES #####################################################################

##### Continuous Outcome PDF ##############################################

class DiceContin_PDF:
    """ Turn a histogram into a probability curve """
    def __init__( self, Nbins = _DEFAULT_DIV ):
        """ Setup to build histo """
        self.Ndat: int         = 0
        self.Nbin: int         = Nbins
        self.wdth: float       = 0.0
        self.data: list[float] = deque()
        self.bnds: list[float] = [0.0 for _ in range( self.Nbin )]
        self.bins: list[int]   = [0   for _ in range( self.Nbin )]
        self.curv: list[float] = [0.0 for _ in range( self.Nbin )]
        self.prob: list[float] = [0.0 for _ in range( self.Nbin )]


    def set_data( self, data : list[float] ):
        """ Store, Sort, and Count """
        self.data = list( data )
        self.data.sort()
        self.Ndat = len( self.data )
        vMin = self.data[0]
        vMax = self.data[-1]
        span = vMax - vMin
        self.wdth = span / self.Nbin
        # wdth = span / (self.Nbin-1)
        for i in range( 1, self.Nbin+1 ):
            self.bnds[i-1] = vMin + i * self.wdth
        # for i in range( self.Nbin+1 ):
        #     self.bnds[i] = vMin + i * wdth            
        j = 0
        for datum in self.data:
            while self.bnds[j] < datum:
                j += 1
            if datum <= self.bnds[j]:
                self.bins[j] += 1
            else:
                raise ValueError( "`set_data()`: THIS SHOULD NOT HAVE HAPPENED!" )
        self.curv = (np.array( self.bins ) / self.Ndat).tolist()
        total = 0.0
        for i, prob_i in enumerate( self.curv ):
            total += prob_i
            self.prob[i] = total


    @staticmethod
    def chop_tail( binPopLst : list[int] ) -> list[int]:
        """ Return a version of `binPopLst` without the long tail """
        binPopLst = deque( binPopLst )
        # binPopLst.pop()
        limit = 2
        Nzero = 0
        while (binPopLst[-1] == 0) or (Nzero < limit):
            if binPopLst[-1] > 0:
                Nzero += 1
            binPopLst.pop()
        return list( binPopLst )
    

    def fit_lognorm_to_data( self ):
        shape, loc, scale = lognorm.fit( chop_sigmas( self.data, 3.0, useMean = False ), floc = 0 )
        print( f"Log-Normal Fit - Shape: {shape}, Location: {loc}, Scale: {scale}" )


    def fit_poisson_to_curv( self ):

        def fit_function( k, lamb ):
            '''poisson function, parameter lamb is the fit parameter'''
            return poisson.pmf( k, lamb )
        
        entries     = self.chop_tail( self.bins )
        bin_centers = [float(self.bnds[i]-self.wdth/2.0) for i in range( len( entries ) )]
        # bin_centers = self.bnds
        # print( bin_centers )
        print( entries )

        # fit with curve_fit
        parameters, cov_matrix = curve_fit( fit_function, bin_centers, entries )
        
        print( "##### Poisson Fit #####" )
        print( parameters )
        print( cov_matrix )



    def sample_value( self ):
        """ Sample from a discrete distribution """
        uniform = random()
        for i, bound in enumerate( self.prob ):
            if uniform <= bound:
                # return self.bnds[max(i-1,0)]
                return self.bnds[i] - self.wdth/2.0
        return self.bnds[-1] - self.wdth/2.0
        

    def save( self, path : str ):
        """ Save enough data to restore the dice roll """
        with open( path, 'w' ) as f:
            json.dump( {
                "bins"  : self.bins,
                "bounds": self.bnds,
                "prob"  : self.prob,
            }, f, indent = 2 )


    @staticmethod
    def load( path : str ):
        """ Load enough data to restore the dice roll """
        rtnObj = DiceContin_PDF()
        with open( path, 'r' ) as f:
            data = json.load(f)
            rtnObj.bins = data["bins"  ]
            rtnObj.bnds = data["bounds"]
            rtnObj.prob = data["prob"  ]
        return rtnObj


##### Binary Outcome CDF ##################################################

class DiceBinary_CDF:
    """ Use a CDF to roll for a binary outcome """
    def __init__( self, Xval : list[float], Yprb : list[float] ):
        """ Store CDF """
        self.valu = list( Xval )
        self.prob = list( Yprb )


    def sample_outcome( self, val : float ) -> bool:
        """ Sample from a discrete probability at the given `val`ue """
        # ASSUMPTION: VALUES ARE CLOSE ENOUGH TOGETHER TO FAITHFULLY REPRESENT THE OUTCOME PROBABILITY 
        pPos = 0.0
        for i, v in self.valu:
            pPos = self.prob[i]
            if v >= val:
                break
        return (random <= pPos)
    

    def save( self, path : str ):
        """ Save enough data to restore the dice roll """
        with open( path, 'w' ) as f:
            json.dump( {
                "value": self.valu,
                "prob" : self.prob,
            }, f, indent = 2 )


    @staticmethod
    def load( path : str ):
        """ Load enough data to restore the dice roll """
        rtnObj = DiceBinary_CDF()
        with open( path, 'r' ) as f:
            data = json.load(f)
            rtnObj.valu = data["value"]
            rtnObj.prob = data["prob" ]
        return rtnObj



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

    _JSON_PATH = {
        "Overall Posn Err" : "json/OverallPosnErr.json",
        "Failure-v-Err_CDF": "json/FailVErr_CDF.json",
    }