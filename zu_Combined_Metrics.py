########## INIT ####################################################################################
import pickle, os, gc, traceback, json

from collections import deque
from typing import Any, Deque
from pprint import pprint
from random import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from aspire.symbols import GraspObj, ObjPose, extract_position, euclidean_distance_between_symbols
from aspire.BlocksTask import set_blocks_env
from aspire.env_config import env_var
from TaskPlanner import set_experiment_env
from draw_beliefs import set_render_env
from magpie_control.realsense_wrapper import MPCD
from Reader import EROM_Reader

from homog_utils import homog_xform, diff_mag



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


##### Environment && Constants ############################################
set_blocks_env()
set_experiment_env()
set_render_env()


########## TYPES ###################################################################################

class PlanStat( Enum ):
    OKAY        = 0
    FAIL        = 1
    COLLIDE     = 2
    BLC_MISSING = 3



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


    def sample_outcome( self, val : float ):
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

        

########## MAIN ####################################################################################

_GET_STATS = False
_EP_EVENTS = False
_CONFUSION = False
_THINIFY   = True

_MISC_DIR = "/media/james/STARGAZER/DATA_TANK/misc_data/" 
_SIM_INFO_PATH = f"{_MISC_DIR}SimInfo.pkl" 

_JSON_PATH = {
    "Overall Posn Err" : "json/OverallPosnErr.json"
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

                make_histo( Xe_t, f"POS ERR, {setNam}:{test}" , xLabel = 'Err', yLabel = 'Occurrences', savefig = True )
                roll = DiceContin_PDF( Nbins = _DEFAULT_DIV )
                roll.set_data( Xe_t )
                roll.save( f"json/PsnErr.{setNam}.{test}.json" )

                Xr = deque()
                for _ in range( 1000000 ):
                    Xr.append( roll.sample_value() )
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
            failDens.append( (ea[0], totlFail/N_fail,) )
            pErrDens.append( (ea[0], st/N_all,) )

        Xe = [item[0] for item in failDens]
        Ya = [item[1] for item in failDens]
        Yp = [item[1] for item in pErrDens]

        xy_plot_filled_under( Xe, Ya, plotTitle = f"ALL TESTS", xLabel = "Err", yLabel = "Cumul. Prob." )

        Nx = len( Xe )
        make_histo( Xe, f"POS ERR, Actual" , xLabel = 'Err', yLabel = 'Occurrences', savefig = True )
        roll = DiceContin_PDF( Nbins = _DEFAULT_DIV )
        roll.set_data( Xe )
        roll.save( _JSON_PATH["Overall Posn Err"] )

        Xr = deque()
        for _ in range( 1000000 ):
            Xr.append( roll.sample_value() )
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

                ### For every episode ###
                for _i_, episodePath in enumerate( testRecord ):
                    print( f"\n{episodePath}, {int(os.path.getsize(episodePath)/1e6)}MB" )
                    try:
                        reader = EROM_Reader( episodePath, suppressLoad = True )
                    except RuntimeError:
                        print( f"\nSKIPPED: {episodePath}\n" )
                        continue

                    resDct_i = reader.planning_failure_vs_hallucination()
                    resDct_i["episode"] = _i_+1
                    resDct_i.update( reader.action_failure_vs_position_variation() )
                    totRes[ setNam ][ test ].append( resDct_i )

        with open( _SIM_INFO_PATH, 'wb' ) as f:
            pickle.dump( totRes, f )

    except KeyboardInterrupt:
        print( "\nSESSION ENDED BY USER!\n" )


    # print( f"\n\n{totalBad}/{totalSteps} = {totalBad/totalSteps*100.0}% BAD PLANNING ATTEMPTS\n\n" )
                

if _THINIFY:
    reader.thinify_recordings_as_files()
    reader.erase() # Flush main recording from memory

    epPrefix = f"{episodePath}".replace( ".pkl", "" )

    statePaths = [item for item in trueRecord if ((epPrefix in f"{item}") and ("_OCV-State" in f"{item}") and (os.path.getsize(item) >= _MIN_STATE_SIZE_BYTES))    ]
    statePaths.sort( key = lambda x: dex_key( x ) )

    for _j_, sPath in enumerate( statePaths ):
        print_header( f"STATE {_j_ + 1}, {sPath}", preWidth = 5, totWidth = 75, capitalize = False )

        reader.thinify_state_file( sPath )



########## EXIT ####################################################################################
crash_out( notify = False )