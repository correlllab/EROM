########## INIT ####################################################################################
import pickle, os, gc, traceback, json

from collections import deque
from typing import Any, Deque
from pprint import pprint
from random import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson, lognorm

from aspire.symbols import GraspObj, ObjPose, extract_position, euclidean_distance_between_symbols
from aspire.BlocksTask import set_blocks_env
from aspire.env_config import env_var
from TaskPlanner import set_experiment_env
from draw_beliefs import set_render_env
from magpie_control.realsense_wrapper import MPCD
from Reader import EROM_Reader

from analysis_Utils import DiceContin_PDF, DiceBinary_CDF, fit_lognorm_to_data


##### Environment && Constants ############################################
set_blocks_env()
set_experiment_env()
set_render_env()


########## CONSTANTS ###############################################################################

_DATA_DRIVE = "STARGAZER/DATA_TANK"

_PLOT_DIR   = "/media/james/FILEPILE/EROM/data/plots/"
_GC_CYCLE   = False 
_F_EXTRACT  = f"{_PLOT_DIR}outData.pkl"
_T_EXTRACT  = f"{_PLOT_DIR}outText.json"

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


########## HELPER CLASSES ##########################################################################

class ConfMatx:
    """ Class for building a Confusion Matrix """
    def __init__( self ):
        self.N_tot : int  = 0
        self.labels: list[str]  = list()
        self.matx  : np.ndarray = None


    def add_class( self, label ):
        """ Add a new label to the list of classes, In order """
        self.labels.append( label )


    def add_classes( self, nuLabels ):
        """ Add a list of new labels to the list of classes, In order """
        if isinstance( nuLabels, (list, deque, tuple) ):
            self.labels.extend( nuLabels )
        else:
            raise TypeError( f"Additional labels were NOT iterable!: {type(nuLabels)}, {nuLabels}" )


    def get_index( self, label ):
        """ Get the row index of the label, Throw when not found """
        return self.labels.index( label )


    def init_matx( self ):
        """ Get ready to count! """
        rowsN = len( self.labels )
        self.matx  = np.zeros( (rowsN,rowsN,) )


    def count_example( self, actual : str, predicted : str ):
        """ Add an example to the `matx`, to be normalized later """
        i = self.get_index( actual    )
        j = self.get_index( predicted )
        self.matx[i,j] += 1
        self.N_tot     += 1


    def match( self, actual : Deque[GraspObj], predicted : Deque[GraspObj], thresh = None ):
        """ Get a matching of `predicted` items to `actual` items """
        # ASSUMPTION: THE LAST LABEL IS THE "NULL LABEL"
        if thresh is None:
            thresh = env_var( "_BAYES_RAD_L2_M" )

        noneLabel: str                   = self.labels[-1] 
        matches  : Deque[tuple[str,str]] = deque()
        matchSet : set[int]              = set([])

        for i, pred_i in enumerate( predicted ):
            dMin = 6e10
            sMin = None
            for actl_j in actual:
                d_ij = euclidean_distance_between_symbols( pred_i, actl_j )
                if (d_ij < dMin) and (d_ij <= thresh):
                    if id(actl_j) not in matchSet:
                        dMin = d_ij
                        sMin = actl_j
            if sMin is not None:
                matchSet.add( id(sMin) )
                matches.append( (sMin.label, pred_i.label,) )
            else:
                matches.append( (noneLabel, pred_i.label,) )

        missing = 0
        for actl_j in actual:
            if id(actl_j) not in matchSet:
                missing += 1
                matches.append( (actl_j.label, noneLabel,) )

        found = 3 - missing
        if found > 0:
            matches.extend( [ (noneLabel, noneLabel,) for _ in range( found ) ] )

        return matches


    def count_examples( self, actual : Deque[GraspObj], predicted : Deque[GraspObj] ):
        """ Count examples from this state """
        matches = self.match( actual, predicted )
        for m in matches:
            self.count_example( *m )
    

    def normalize( self ):
        """ Normalize the confusion matrix """
        # NOTE: Normalizing twice should (probably) NOT have any bad effects!
        rowsN = len( self.labels )
        nMtx  = np.zeros( (rowsN,rowsN,) )

        for i, row in enumerate( self.matx ):
            N_i = np.sum( row )
            nMtx[i,:] = row / N_i

        self.matx = nMtx.copy()


    def get_matx( self, normalizeM = False ):
        """ Return a copy of the matx """
        if normalizeM:
            self.normalize()
        return self.matx.copy()
    
"""
########## TEST, RGB: KC-KP #####################################################################
Search Times
Log-Normal Fit - Shape: 0.28674900506295525, Location: 0, Scale: 93.90954640387551
Action Times
Log-Normal Fit - Shape: 0.05144232369223114, Location: 0, Scale: 18.697605905294804


########## TEST, RGB: SC-KP #####################################################################
Search Times
Log-Normal Fit - Shape: 0.47115397552083116, Location: 0, Scale: 82.87279800944911
Action Times
Log-Normal Fit - Shape: 1.335055424877319, Location: 0, Scale: 8.767471695780298


########## TEST, RGB: KC-SP #####################################################################
Search Times
Log-Normal Fit - Shape: 0.44645100101638924, Location: 0, Scale: 69.1421987530391
Action Times
Log-Normal Fit - Shape: 0.7665100421157891, Location: 0, Scale: 11.93487992351694


########## TEST, RGB: SC-SP #####################################################################
Search Times
Log-Normal Fit - Shape: 0.4954870103662892, Location: 0, Scale: 74.24385054021829
Action Times
Log-Normal Fit - Shape: 1.4953473899683347, Location: 0, Scale: 7.9200927749431385


########## TEST, RBW: KC-KP #####################################################################
Search Times
Log-Normal Fit - Shape: 0.23601114474261797, Location: 0, Scale: 78.44404936626925
Action Times
Log-Normal Fit - Shape: 0.07387960239384964, Location: 0, Scale: 21.202165368135727


########## TEST, RBW: SC-KP #####################################################################
Search Times
Log-Normal Fit - Shape: 0.4773927781806837, Location: 0, Scale: 107.1830264822935
Action Times
Log-Normal Fit - Shape: 0.8069032963493088, Location: 0, Scale: 11.356399726082188


########## TEST, RBW: KC-SP #####################################################################
Search Times
Log-Normal Fit - Shape: 0.4620953772377743, Location: 0, Scale: 101.7941468753413
Action Times
Log-Normal Fit - Shape: 0.7891396574154087, Location: 0, Scale: 11.567504797464126


########## TEST, RBW: SC-SP #####################################################################
Search Times
Log-Normal Fit - Shape: 0.5012157834032029, Location: 0, Scale: 107.11724660211397
Action Times
Log-Normal Fit - Shape: 0.800120382970747, Location: 0, Scale: 10.739329853228101

"""


"""
########## TEST, RGB: KC-KP, N_EP: 20 ###########################################################
P(Plan)    = 0.75
P(Missing) = 0.0
P(Plan|3)  = 0.75

P(Plan | 0 Halluc) = 0.75
P(0 Halluc), [N=80] = 1.0

POS ERR, RGB:KC-KP
Mean: ___ 0.007634486667450442
Median: _ 0.0067866851143173704
Std.Dev.: 0.005232707102501245
Log-Normal Fit - Shape: 0.3328237189743045, Location: 0, Scale: 0.006732495552602966

POS ERR, Simulated, RGB:KC-KP
Mean: ___ 0.007621337477333922
Median: _ 0.006515029438909591
Std.Dev.: 0.005190748716177473



########## TEST, RGB: SC-KP, N_EP: 21 ###########################################################
P(Plan)    = 0.7180616740088106
P(Missing) = 0.21585903083700442
P(Plan|3)  = 0.896774193548387

P(Plan | 0 Halluc) = 0.7156862745098039
P(0 Halluc), [N=204] = 0.8986784140969163

P(Plan | 1 Halluc) = 0.75
P(1 Halluc), [N=20] = 0.0881057268722467

P(Plan | 2 Halluc) = 0.6666666666666666
P(2 Halluc), [N=3] = 0.013215859030837005

POS ERR, RGB:SC-KP
Mean: ___ 0.008045120557593075
Median: _ 0.006633071491109657
Std.Dev.: 0.008776425214217252
Log-Normal Fit - Shape: 0.4110499978073131, Location: 0, Scale: 0.006529559805858347

POS ERR, Simulated, RGB:SC-KP
Mean: ___ 0.008039988740516377
Median: _ 0.006794683239167132
Std.Dev.: 0.00872126730821331



########## TEST, RGB: KC-SP, N_EP: 19 ###########################################################
P(Plan)    = 0.6446280991735537
P(Missing) = 0.25206611570247933
P(Plan|3)  = 0.8690476190476191

P(Plan | 0 Halluc) = 0.6506550218340611
P(0 Halluc), [N=229] = 0.9462809917355371

P(Plan | 1 Halluc) = 0.4
P(1 Halluc), [N=10] = 0.04132231404958678

P(Plan | 2 Halluc) = 1.0
P(2 Halluc), [N=2] = 0.008264462809917356

P(Plan | 4 Halluc) = 1.0
P(4 Halluc), [N=1] = 0.004132231404958678

POS ERR, RGB:KC-SP
Mean: ___ 0.008981807078503047
Median: _ 0.007251370777023036
Std.Dev.: 0.010867777864416558
Log-Normal Fit - Shape: 0.509699407852857, Location: 0, Scale: 0.00684499430538845

POS ERR, Simulated, RGB:KC-SP
Mean: ___ 0.008986072361462583
Median: _ 0.007493969980881261
Std.Dev.: 0.010894788308353673



########## TEST, RGB: SC-SP, N_EP: 20 ###########################################################
P(Plan)    = 0.7791164658634538
P(Missing) = 0.15261044176706828
P(Plan|3)  = 0.9230769230769231

P(Plan | 0 Halluc) = 0.7818181818181819
P(0 Halluc), [N=220] = 0.8835341365461847

P(Plan | 1 Halluc) = 0.75
P(1 Halluc), [N=24] = 0.0963855421686747

P(Plan | 2 Halluc) = 0.8
P(2 Halluc), [N=5] = 0.020080321285140562

POS ERR, RGB:SC-SP
Mean: ___ 0.007336683198845072
Median: _ 0.006805519897754429
Std.Dev.: 0.004801049504029809
Log-Normal Fit - Shape: 0.4368183980151973, Location: 0, Scale: 0.006472601041195895

POS ERR, Simulated, RGB:SC-SP
Mean: ___ 0.007326292964594586
Median: _ 0.006954766749022283
Std.Dev.: 0.004794691643030675



########## TEST, RBW: KC-KP, N_EP: 22 ###########################################################
P(Plan)    = 0.7472527472527473
P(Missing) = 0.02197802197802198
P(Plan|3)  = 0.7640449438202247

P(Plan | 0 Halluc) = 0.7472527472527473
P(0 Halluc), [N=91] = 1.0

POS ERR, RBW:KC-KP
Mean: ___ 0.00799256563731155
Median: _ 0.007270492815854862
Std.Dev.: 0.006397293696278098
Log-Normal Fit - Shape: 0.3325287013874546, Location: 0, Scale: 0.00698380733354094

POS ERR, Simulated, RBW:KC-KP
Mean: ___ 0.00798397401195767
Median: _ 0.007482474242611344
Std.Dev.: 0.0063783635277502755



########## TEST, RBW: SC-KP, N_EP: 20 ###########################################################
P(Plan)    = 0.6056338028169014
P(Missing) = 0.3192488262910798
P(Plan|3)  = 0.84

P(Plan | 0 Halluc) = 0.5699481865284974
P(0 Halluc), [N=193] = 0.9061032863849765

P(Plan | 1 Halluc) = 0.95
P(1 Halluc), [N=20] = 0.09389671361502347

POS ERR, RBW:SC-KP
Mean: ___ 0.00681935023923729
Median: _ 0.0068557182386246715
Std.Dev.: 0.0024817344186230388
Log-Normal Fit - Shape: 0.359772179839813, Location: 0, Scale: 0.006597263556057573

POS ERR, Simulated, RBW:SC-KP
Mean: ___ 0.006815313249237454
Median: _ 0.006799477195276209
Std.Dev.: 0.0024782892388474057



########## TEST, RBW: KC-SP, N_EP: 20 ###########################################################
P(Plan)    = 0.5574712643678161
P(Missing) = 0.3218390804597701
P(Plan|3)  = 0.8333333333333334

P(Plan | 0 Halluc) = 0.5548780487804879
P(0 Halluc), [N=164] = 0.9425287356321839

P(Plan | 1 Halluc) = 0.5714285714285714
P(1 Halluc), [N=7] = 0.040229885057471264

P(Plan | 2 Halluc) = 0.6666666666666666
P(2 Halluc), [N=3] = 0.017241379310344827

POS ERR, RBW:KC-SP
Mean: ___ 0.006187498432133404
Median: _ 0.006343940866243577
Std.Dev.: 0.0023940557029869217
Log-Normal Fit - Shape: 0.429565789953105, Location: 0, Scale: 0.005964496613862417

POS ERR, Simulated, RBW:KC-SP
Mean: ___ 0.006188034037267077
Median: _ 0.006326835102204234
Std.Dev.: 0.002393088170384638



########## TEST, RBW: SC-SP, N_EP: 20 ###########################################################
P(Plan)    = 0.625
P(Missing) = 0.3387096774193548
P(Plan|3)  = 0.9025974025974026

P(Plan | 0 Halluc) = 0.6218487394957983
P(0 Halluc), [N=238] = 0.9596774193548387

P(Plan | 1 Halluc) = 0.7
P(1 Halluc), [N=10] = 0.04032258064516129

POS ERR, RBW:SC-SP
Mean: ___ 0.007175135036980465
Median: _ 0.007461389319378849
Std.Dev.: 0.002491628136670465
Log-Normal Fit - Shape: 0.4262289767155447, Location: 0, Scale: 0.006856369497348779

POS ERR, Simulated, RBW:SC-SP
Mean: ___ 0.007177594934024564
Median: _ 0.0075202090173931814
Std.Dev.: 0.002483770662137547

POS ERR, Actual
Mean: ___ 0.0075287555387563675
Median: _ 0.006884854778681009
Std.Dev.: 0.0064114227005904936

POS ERR, Simulated
Mean: ___ 0.0075483819094679236
Median: _ 0.006794683239167132
Std.Dev.: 0.006483282207430517



"""
        

########## MAIN ####################################################################################
_POSE_SIGMA_CHOP = 2.5
_ACTN_SIGMA_CHOP = 2.5
_SRCH_SIGMA_CHOP = 2.5

_GET_STATS = True
_EP_EVENTS = True
_CONFUSION = False
_THINIFY   = False

_SHOW_PLOT = False

_MISC_DIR = "/media/james/STARGAZER/DATA_TANK/misc_data/" 
_SIM_INFO_PATH = f"{_MISC_DIR}SimInfo.pkl" 

_JSON_PATH = {
    "Overall Posn Err" : "json/OverallPosnErr.json",
    "Failure-v-Err_CDF": "json/FailVErr_CDF.json",
}


if _EP_EVENTS:
    try:
        totRes    = dict() # Input Data 
        totPrb    = dict() # Output Metrics
        totErrAct = deque()
        
        with open( _SIM_INFO_PATH, 'rb' ) as f:
            totRes = pickle.load(f)

        ### For every block set ###
        for iii, paths in enumerate( datasets ):

            setNam = dataLabels[iii]
            suffix = "_" + setNam
            skip   = False

            ### For every scenario ###
            for ii, test in enumerate( tests ):
            # for ii, test in enumerate( tests[1:] ):
            # for ii, test in enumerate( tests[3:] ):
                
                episodes = totRes[ setNam ][ test ] 

                print_header( f"TEST, {setNam}: {test}, N_ep: {len( episodes )}", preWidth = 10, totWidth = 100, capitalize = True )

                ##### What influence do hallucinated blocks have on planning failure? ##############

                testRes = {
                    "resPlan"  : deque(),
                    "N_halluc" : deque(),
                    "N_missng" : deque(),
                    "3 Blocks" : deque(),
                    "P(p|mis)" : deque(),
                    "<Err,Act>": deque(),
                    "P(p|n_H)" : dict(),
                }

                for ep_jj in episodes:
        
                    # P(Plan)
                    # P(N Halluc)

                    # P(Plan | N Halluc)

                    # print( list( ep_jj.keys() ) ) # 'N_halluc', 'N_missng', 'resPlan', 'episode', 'avgErr', 'resActn'
                    X_halluc = ep_jj['N_halluc']
                    Y_plannd = ep_jj['resPlan' ]
                    Z_missng = ep_jj['N_missng']
                    D_posErr = ep_jj['avgErr'  ]
                    A_result = ep_jj['resActn' ]

                    for I, step_I in enumerate( X_halluc ):
                        testRes["N_halluc" ].append( X_halluc[I] )
                        testRes["resPlan"  ].append( Y_plannd[I] )
                        testRes["N_missng" ].append( Z_missng[I] )
                        
                        if D_posErr[I] >= 0.0:
                            testRes["<Err,Act>"].append( (D_posErr[I], A_result[I],) )
                            totErrAct.append( (D_posErr[I], A_result[I],) )

                        if X_halluc[I] not in testRes["P(p|n_H)"]:
                            testRes["P(p|n_H)"][ X_halluc[I] ] = deque()

                        testRes["P(p|n_H)"][ X_halluc[I] ].append( Y_plannd[I] )

                        if (X_halluc[I] == 0) and (Z_missng[I] == 0):
                            testRes["3 Blocks"].append( Y_plannd[I] )

                        if (Z_missng[I] > 0):
                            testRes["P(p|mis)"].append( Y_plannd[I] )

                ##### Probability Metrics #################################

                pos = [ item for item in testRes['resPlan' ] if (item == True) ]
                po3 = [ item for item in testRes['3 Blocks'] if (item == True) ]
                mis = [ item for item in testRes['N_missng'] if (item > 0)     ]
                print( f"P(Plan)    = {len( pos )/len( testRes['resPlan' ] )}" )
                print( f"P(Missing) = {len( mis )/len( testRes['resPlan' ] )}" )
                print( f"P(Plan|3)  = {len( po3 )/len( testRes['3 Blocks'] )}" )
                
                
                N_h = sorted( list( testRes["P(p|n_H)"].keys() ) )
                for k in N_h:
                    pos_k = [ item for item in testRes["P(p|n_H)"][k] if (item == True)]
                    print()
                    print( f"P(Plan | {k} Halluc) = {len( pos_k )/len( testRes['P(p|n_H)'][k] )}" )
                    print( f"P({k} Halluc), [N={len( testRes['P(p|n_H)'][k] )}] = {len( testRes['P(p|n_H)'][k] )/len( testRes['resPlan'] )}" )
        

                ##### What influence does Position Variation have on action failure? ###############

                errAct = list( testRes["<Err,Act>"] )
                errAct.sort( key = lambda x: x[0] )
                
                Xe_t = [item[0] for item in errAct]
                Ya_t = [item[1] for item in errAct]

                if _SHOW_PLOT:
                    make_histo( Xe_t, f"POS ERR, {setNam}:{test}" , xLabel = 'Err', yLabel = 'Occurrences', savefig = True )
                roll = DiceContin_PDF( Nbins = _DEFAULT_DIV )
                roll.set_data( Xe_t )

                roll.fit_lognorm_to_data( chopSigma = _POSE_SIGMA_CHOP )
                
                roll.save( f"json/PsnErr.{setNam}.{test}.json" )

                Xr = deque()
                for _ in range( 1000000 ):
                    Xr.append( roll.sample_value() )
                if _SHOW_PLOT:
                    make_histo( Xr, f"POS ERR, Simulated, {setNam}:{test}" , xLabel = 'Err', yLabel = 'Occurrences', savefig = True )
                
                N_fail = 0
                for ea in errAct:
                    if ea[1] == False:
                        N_fail += 1

        
        totErrAct = list( totErrAct )
        totErrAct.sort( key = lambda x: x[0] )
        N_fail = 0
        N_all  = len(totErrAct)
        for ea in totErrAct:
            if ea[1] == False:
                N_fail += 1

        totlFail = 0                        
        failDens = deque()
        pErrDens = deque()
        
        for st, ea in enumerate( totErrAct ):
            if ea[1] == False:
                totlFail += 1
            if N_fail:
                failDens.append( (ea[0], totlFail/N_fail,) )
            if N_all:
                pErrDens.append( (ea[0], st/N_all,) )

        Xe = [item[0] for item in failDens]
        Ya = [item[1] for item in failDens]
        Yp = [item[1] for item in pErrDens]

        if _SHOW_PLOT:
            xy_plot_filled_under( Xe, Ya, plotTitle = f"ALL TESTS", xLabel = "Err", yLabel = "Cumul. Prob." )

        Nx = len( Xe )
        if _SHOW_PLOT:
            make_histo( Xe, f"POS ERR, Actual" , xLabel = 'Err', yLabel = 'Occurrences', savefig = True )
        roll = DiceContin_PDF( Nbins = _DEFAULT_DIV )
        roll.set_data( Xe )
        roll.save( _JSON_PATH["Overall Posn Err"] )

        cdf = DiceBinary_CDF( Xe, Ya )
        cdf.save( _JSON_PATH["Failure-v-Err_CDF"] )

        Xr = deque()
        for _ in range( 1000000 ):
            Xr.append( roll.sample_value() )
        if _SHOW_PLOT:
            make_histo( Xr, f"POS ERR, Simulated" , xLabel = 'Err', yLabel = 'Occurrences', savefig = True )
        
        

    except KeyboardInterrupt:
        print( "\nSESSION ENDED BY USER!\n" )


"""
['redBlock', 'grnBlock', 'bluBlock', 'NOTHING']

[[0.81077  0.0087336 0.034934 0.14556]
 [0.018576 0.74149   0.17028  0.069659]
 [0.037179 0.12051   0.76282  0.079487]
 [0.041032 0.069949  0.034388 0.85463]]

 
['redBlock', 'blkBlock', 'whtBlock', 'NOTHING']

[[0.73793  0.046552 0.055172  0.16034]
 [0.045388 0.75988  0.0087848 0.18594]
 [0.027451 0.015686 0.72745   0.22941]
 [0.059389 0.034934 0.10175   0.80393]]
"""


if _CONFUSION:

    try:

        ### For every block set ###
        for iii, paths in enumerate( datasets ):

            confMatx = ConfMatx()

            setNam = dataLabels[iii]
            suffix = "_" + setNam
            skip   = False

            print( eBlcNam[ setNam ] )
            confMatx.add_classes( eBlcNam[ setNam ] )
            confMatx.init_matx()

            totRes[ setNam ] = dict()

            ### For every scenario ###
            for ii, test in enumerate( tests ):
                
                totRes[ setNam ][ test ] = deque()

                print_header( f"TEST, {setNam}: {test}", preWidth = 10, totWidth = 100, capitalize = True )

                ##### Init ####################################################
                path     = paths[ii]
                longTNam = longTestNames[ii]

                testRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" not in f"{item}") and ("thin" not in f"{item}".lower()))]
                trueRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" in f"{item}")     and ("thin" not in f"{item}".lower()))]

                ### For every episode ###
                for _i_, episodePath in enumerate( testRecord ):
                    # print( f"\n{episodePath}, {int(os.path.getsize(episodePath)/1e6)}MB" )
                    try:
                        reader = EROM_Reader( episodePath, suppressLoad = True )
                    except RuntimeError:
                        print( f"\nSKIPPED: {episodePath}\n" )
                        continue

                    reader.count_into_confusion_matrix( confMatx )

            confMatx.normalize()

            print( confMatx.get_matx() )

    except KeyboardInterrupt:
        print( "\nSESSION ENDED BY USER!\n" )

if _GET_STATS:

    totalBad = 0
    totalSteps = 0
    totRes = dict()

    try:
        ### For every block set ###
        for iii, paths in enumerate( datasets ):

            setNam = dataLabels[iii]
            suffix = "_" + setNam
            skip   = False

            totRes[ setNam ] = dict()


            ### For every scenario ###
            for ii, test in enumerate( tests ):
            # for ii, test in enumerate( tests[1:] ):
            # for ii, test in enumerate( tests[3:] ):
                
                totRes[ setNam ][ test ] = deque()

                print_header( f"TEST, {setNam}: {test}", preWidth = 10, totWidth = 100, capitalize = True )

                ##### Init ####################################################
                path     = paths[ii]
                longTNam = longTestNames[ii]

                testRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" not in f"{item}") and ("thin" not in f"{item}".lower()))]
                trueRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" in f"{item}")     and ("thin" not in f"{item}".lower()))]

                tSearch = deque()
                tAction = deque()

                ### For every episode ###
                for _i_, episodePath in enumerate( testRecord ):
                    # print( f"\n{episodePath}, {int(os.path.getsize(episodePath)/1e6)}MB" )
                    try:
                        reader = EROM_Reader( episodePath, suppressLoad = True )
                    except RuntimeError:
                        print( f"\nSKIPPED: {episodePath}\n" )
                        continue

                    resDct_i = reader.planning_failure_vs_hallucination()
                    resDct_i["episode"] = _i_+1
                    resDct_i.update( reader.action_failure_vs_position_variation() )
                    totRes[ setNam ][ test ].append( resDct_i )

                    # reader.print_all_step_msgs()

                    tSearch_i = reader.aggregate_search_times()
                    tSearch.extend( tSearch_i )

                    tAction_i = reader.aggregate_action_times()
                    tAction.extend( tAction_i )
                
                tSearch = list( tSearch )
                muSrch  = np.mean( tSearch )
                mdSrch  = np.median( tSearch )
                stSrch  = np.std( tSearch ) 

                print( "Search Times" )
                fit_lognorm_to_data( tSearch, sigmaChop = _SRCH_SIGMA_CHOP )
                # print( f"Mean: ____ {muSrch}" )
                # print( f"Median: __ {mdSrch}" )
                # print( f"Std. Dev.: {stSrch}\n" )

                tAction = list( tAction )
                muActn  = np.mean( tAction )
                mdActn  = np.median( tAction )
                stActn  = np.std( tAction ) 

                print( "Action Times" )
                fit_lognorm_to_data( tAction, sigmaChop = _ACTN_SIGMA_CHOP )
                # print( f"Mean: ____ {muActn}" )
                # print( f"Median: __ {mdActn}" )
                # print( f"Std. Dev.: {stActn}\n" )


        with open( _SIM_INFO_PATH, 'wb' ) as f:
            pickle.dump( totRes, f )

    except KeyboardInterrupt:
        print( "\nSESSION ENDED BY USER!\n" )


    # print( f"\n\n{totalBad}/{totalSteps} = {totalBad/totalSteps*100.0}% BAD PLANNING ATTEMPTS\n\n" )
                

if _THINIFY:

    ### For every block set ###
    for iii, paths in enumerate( datasets ):

        setNam = dataLabels[iii]
        suffix = "_" + setNam
        skip   = False

        ### For every scenario ###
        for ii, test in enumerate( tests ):

            print_header( f"TEST, {setNam}: {test}", preWidth = 10, totWidth = 100, capitalize = True )

            ##### Init ####################################################
            path     = paths[ii]
            longTNam = longTestNames[ii]

            testRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" not in f"{item}") and ("thin" not in f"{item}".lower()))]
            trueRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" in f"{item}")     and ("thin" not in f"{item}".lower()))]

            ### For every episode ###
            for _i_, episodePath in enumerate( testRecord ):
                # print( f"\n{episodePath}, {int(os.path.getsize(episodePath)/1e6)}MB" )
                try:
                    reader = EROM_Reader( episodePath, suppressLoad = False )
                except RuntimeError:
                    print( f"\nSKIPPED: {episodePath}\n" )
                    continue

                reader.thinify_recordings_as_files()
                reader.erase() # Flush main recording from memory

                epPrefix = f"{episodePath}".replace( ".pkl", "" )

                statePaths = [item for item in trueRecord if ((epPrefix in f"{item}") and ("_OCV-State" in f"{item}") and ("THIN" not in f"{item}") and (os.path.getsize(item) >= _MIN_STATE_SIZE_BYTES))    ]
                statePaths.sort( key = lambda x: dex_key( x ) )

                for _j_, sPath in enumerate( statePaths ):
                    # print_header( f"STATE {_j_ + 1}, {sPath}", preWidth = 5, totWidth = 75, capitalize = False )

                    reader.thinify_state_file( sPath )



########## EXIT ####################################################################################
crash_out( notify = False )