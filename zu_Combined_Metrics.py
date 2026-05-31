########## INIT ####################################################################################
import pickle, os, gc, traceback, json

from collections import deque

from aspire.symbols import GraspObj
from aspire.BlocksTask import set_blocks_env
from TaskPlanner import set_experiment_env
from draw_beliefs import set_render_env



########## CONSTANTS ###############################################################################

_DATA_DRIVE = "STARGAZER/DATA_TANK"

_PLOT_DIR   = "/media/james/FILEPILE/EROM/data/plots/"
_GC_CYCLE   = False 
_F_EXTRACT  = f"{_PLOT_DIR}outData.pkl"
_T_EXTRACT  = f"{_PLOT_DIR}outText.json"

_MIN_STATE_SIZE_BYTES = 500.0


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


def dex_key( x ):
    dex = f"{x}".split('_')[-1].replace( ".pkl", "" )
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


########## DATA EXTRACTION CLASS ###################################################################

class EROM_Reader:
    """ Class to interpret the recordings I made """

    @staticmethod
    def get_thin_struct_copy( dtmDat ):
        """ Strip all the visual data from the struct and return """

        def recur( datumData ):
            if isinstance( datumData, dict ):
                rtnDct = dict()
                for k, v in datumData.items():
                    rtnDct[k] = recur(v)
                return rtnDct
            elif isinstance( datumData, (list,deque,) ):
                rtnDqu = deque()
                for item in datumData:
                    rtnDqu.append( recur( item ) )
                return list( rtnDqu )
            elif isinstance( datumData, GraspObj ):
                return datumData.copy( thin = True )
            else:
                return datumData
            
        return recur( dtmDat )


    def __init__( self, episodePath  ):
        """ Load the episode """
        self.episodePath : str = episodePath
        self.data        : list = None
        try:
            with open( episodePath, 'rb' ) as f:
                self.data = pickle.load( f )
        except EOFError as e:
            raise RuntimeError( f"LOAD ERROR: {e}" )
        

    def erase( self ):
        self.data = None
        gc.collect()
        

    def split_data_into_steps( self ):
        """ Split the data into steps associated with the state files we made """
        epsdRecord = deque()
        stepRecord = deque()
        started    = False

        for datum in self.data:
            dtmMsg = datum['msg']
            dtmT   = datum['t']
            dtmDat = datum['data']
            
            if (dtmMsg == "BGN: Phase 1") and started:
                if len( stepRecord ):
                    epsdRecord.append( list( stepRecord ) )
                stepRecord = deque()

            if (dtmMsg == "BGN: Phase 1"):
                started = True
            
            if started:
                thnDtm = {
                    'msg' : dtmMsg,               
                    't'   : dtmT,               
                    'data': EROM_Reader.get_thin_struct_copy( dtmDat ),                   
                }
                stepRecord.append( thnDtm )

        if len( stepRecord ):
            epsdRecord.append( stepRecord )

        return list( epsdRecord )


    def thinify_recordings_as_files( self ):
        """ Create files that are faster to process """
        prefix = self.episodePath.split('.')[0]
        infix  = "_Thin-Step_"
        dataSteps = self.split_data_into_steps()
        for _i_, step in enumerate(dataSteps):
            postfix = f"{_i_}"
            outPath = prefix + infix + postfix + ".pkl"
            with open( outPath, 'wb' ) as f:
                pickle.dump( step, f )


    @staticmethod
    def thinify_state_file( fPath : str ):
        try:
            with open( fPath, 'rb' ) as f:
                data = pickle.load( f )
        except EOFError as e:
            raise RuntimeError( f"LOAD ERROR: {e}" )
        
        thinData = EROM_Reader.get_thin_struct_copy( data )
        oPath    = f"{fPath.split('.')[0]}_THIN.pkl"
        with open( oPath, 'wb' ) as f:
            pickle.dump( thinData, f )



########## MAIN ####################################################################################
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

        testRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" not in f"{item}"))]
        trueRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" in f"{item}"))    ]

        ### For every episode ###
        for episodePath in testRecord:
            print( f"\n{episodePath}, {int(os.path.getsize(episodePath)/1e6)}MB" )
            try:
                reader = EROM_Reader( episodePath )
            except RuntimeError:
                print( f"\nSKIPPED: {episodePath}\n" )
                continue

            reader.thinify_recordings_as_files()
            reader.erase() # Flush main recording from memory

            epPrefix = f"{episodePath}".replace( ".pkl", "" )

            statePaths = [item for item in trueRecord if ((epPrefix in f"{item}") and ("_OCV-State" in f"{item}") and (os.path.getsize(item) >= _MIN_STATE_SIZE_BYTES))    ]
            statePaths.sort( key = lambda x: dex_key( x ) )

            for _j_, sPath in enumerate( statePaths ):
                print_header( f"STATE {_j_ + 1}, {sPath}", preWidth = 5, totWidth = 75, capitalize = False )

                reader.thinify_state_file( sPath )

########## EXIT ####################################################################################
crash_out( notify = True )