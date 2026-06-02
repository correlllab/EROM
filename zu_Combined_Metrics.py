########## INIT ####################################################################################
import pickle, os, gc, traceback, json

from collections import deque
from typing import Any

import numpy as np

from aspire.symbols import GraspObj, ObjPose
from aspire.BlocksTask import set_blocks_env
from TaskPlanner import set_experiment_env
from draw_beliefs import set_render_env
from magpie_control.realsense_wrapper import MPCD



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


########## DATA EXTRACTION CLASS ###################################################################

class EROM_Reader:
    """ Class to interpret the recordings I made """


    ##### Data Prep #######################################################

    @staticmethod
    def get_thin_struct_copy( dtmDat ):
        """ Strip all the visual data from the struct and return """

        def recur( datumData ):
            """ Recursively trim away heavy structures """
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
            elif isinstance( datumData, (tuple,) ):
                rtnDqu = deque()
                for item in datumData:
                    rtnDqu.append( recur( item ) )
                return tuple( rtnDqu )
            elif isinstance( datumData, ObjPose ):
                return datumData.pose.tolist()
            elif isinstance( datumData, GraspObj ):
                return datumData.copy( thin = True )
            elif isinstance( datumData, np.ndarray ):
                if isinstance( datumData.size, int ):
                    if datumData.size > 100:
                        return None
                else:
                    for s in datumData.size:
                        if s > 100:
                            return None
                return datumData
            # White List
            elif isinstance( datumData, (str, float, int, np.float32, np.float64, np.int32, np.int64,) ) or ("status" in str(datumData.__class__.__name__).lower()):
                return datumData
            else:
                print( "THROW OUT:", type( datumData ) )
                return None
            
        return recur( dtmDat )


    def __init__( self, episodePath = None, suppressLoad = False ):
        """ Load the episode """
        self.episodePath : str  = episodePath
        self.data        : list = None
        if (episodePath is not None) and (not suppressLoad):
            try:
                with open( episodePath, 'rb' ) as f:
                    self.data = pickle.load( f )
            except EOFError as e:
                raise RuntimeError( f"LOAD ERROR: {e}" )
        

    def erase( self ):
        """ Clear the portion of the heap owned by this object """
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
        prefix    = self.episodePath.split('.')[0]
        infix     = "_Thin-Step_"
        dataSteps = self.split_data_into_steps()
        for _i_, step in enumerate(dataSteps):
            postfix = f"{_i_}" # WARNING: THIS WAS WRONG, ADD 1
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


    ##### Data Processing #################################################

    def get_states_and_steps( self ):
        directory  = os.path.dirname( self.episodePath )
        fileName   = self.episodePath.split('/')[-1]
        prefix     = "_".join( fileName.split('_')[:3] ).replace(".pkl","")

        # 1. Get the (thin) states and the (thin) logs, associated, in order
        assocSteps = sorted( 
            [os.path.join( directory, item ) for item in sorted( os.listdir( directory ) ) if (prefix in f"{item}") and ("Thin-Step" in item)],
            key = lambda x: dex_key(x)
        )
        assocStates = sorted( 
            [os.path.join( directory, item ) for item in sorted( os.listdir( directory ) ) if (prefix in f"{item}") and ("_THIN" in item) and ("e_THIN" not in item)],
            key = lambda x: dex_key(x, offset=-2)
        )

        # print( directory )
        # print( fileName  )
        # print( prefix    )
        # for fName in assocStates:
        #     print( fName )
        # for fName in assocSteps:
        #     print( fName )
        
        return assocStates, assocSteps


    @staticmethod
    def planning_result_from_thin_step( step : list[dict[str,Any]] ):
        """ Return the result of planning """
        for datum in step:
            try:
                if "END: Phase 3" in datum["msg"]:
                    dtmDat = datum["data"]
                    if (dtmDat is not None) and len( dtmDat ):
                        if len( dtmDat["plan"] ):
                            print( dtmDat )
                            return True
                        else:
                            return False
                    elif (dtmDat is None) or (not len( dtmDat )):
                        return None
                    else:
                        raise ValueError( "THIS SHOULD NOT HAPPEN!" )
            except TypeError:
                return None
        return False
    
    
    def planning_failure_vs_hallucination( self ):
        """ What influence do hallucinated blocks have on planning failure? """
        assocStates, assocSteps = self.get_states_and_steps()

        """
        "labels" : list(), #- List of objects in this scene
        "image"  : dict(), #- Lookup of color images used
        "depth"  : dict(), #- Lookup of depth images used
        "clouds" : deque(), # Collection of clouds obtained from the masked images
        "objects": deque(), # Collection of readings obtained from the masked images
        "sensed" : list(), # Collection of symbols obtained from the robot
        "symbols": dict(), #- Lookup of objects obtained from the readings
        """

        totalSteps = len( assocStates )
        totalBad   = 0

        for i in range( len( assocStates ) ):
            fState_i = assocStates[i]
            fStep_i  = assocSteps[i]

            with open( fState_i, 'rb' ) as f:
                state_i = pickle.load(f)
            with open( fStep_i, 'rb' ) as f:
                step_i = pickle.load(f)
                # print( list( state_i.keys() ) ) # ['labels', 'image', 'depth', 'clouds', 'objects', 'sensed', 'symbols']
                
                # print( "\nsensed" )
                

                if len( state_i["sensed"] ) != 3:
                    
                    print()
                    if len( state_i["sensed"] ) > 3:
                        print( "HALLUCINATION" )
                    elif len( state_i["sensed"] ) < 3:
                        print( "MISSING BLOCK" )
                        if EROM_Reader.planning_result_from_thin_step( step_i ):
                            totalBad += 1

                    print( "Sensed" )
                    for object_j in state_i["sensed"]:
                        print( object_j )

                    print( "Symbols" )
                    for name, object_j in state_i["symbols"].items():
                        print( name, object_j )

                    print( "plan:", EROM_Reader.planning_result_from_thin_step( step_i ))
                    print( "actn:", EROM_Reader.action_result_from_thin_step( step_i ))
                    # crash_out()

                # print( "\nobjects" )
                # for object_j in state_i["objects"]:
                #     print( object_j )

                # print( "\nsymbols" )
                # for object_j in state_i["symbols"]:
                #     print( object_j )
            
        print( f"\n\n{totalBad}/{totalSteps} BAD PLANNING ATTEMPTS\n\n" )
        return totalBad, totalSteps

    @staticmethod
    def action_result_from_thin_step( step : list[dict[str,Any]] ):
        """ Return the result of the action """
        for datum in step:
            if "BT END: Status.SUCCESS" in datum["msg"]:
                return True
            if "BT END: Status.FAILURE" in datum["msg"]:
                return False
        return None

    
    def action_failure_vs_position_variation( self ):
        """ What influence does Position Variation have on action failure? """
        
        assocStates, assocSteps = self.get_states_and_steps()

        # 2. Get the position variation of the block to be manipulated
        for i in range( len( assocStates ) ):
            fStep_i  = assocSteps[i]
            fState_i = assocStates[i]
            state_i  = None
            step_i   = None

            with open( fState_i, 'rb' ) as f:
                state_i = pickle.load(f)
                print( state_i["objects"][0].ts )

            with open( fStep_i, 'rb' ) as f:
                step_i = pickle.load(f)
                for datum in step_i:
                    print( datum["t"], datum["msg"] )


            if (state_i is not None) and (step_i is not None):
                stepRes = EROM_Reader.action_result_from_thin_step( step_i )
                print( f"Action Result: {stepRes}" )

                # print( list( state_i.keys() ) )
            break

        # 3. Histogram of action failures in each variation bin


########## MAIN ####################################################################################

_THINIFY   = False
_POSN_ACTN = True

totalBad = 0
totalSteps = 0

try:
    ### For every block set ###
    for iii, paths in enumerate( datasets ):
        
        setNam = dataLabels[iii]
        suffix = "_" + setNam
        skip   = False


        ### For every scenario ###
        for ii, test in enumerate( tests ):
        # for ii, test in enumerate( tests[1:] ):
        # for ii, test in enumerate( tests[3:] ):
            
            print_header( f"TEST, {setNam}: {test}", preWidth = 10, totWidth = 100, capitalize = True )

            ##### Init ####################################################
            path     = paths[ii]
            longTNam = longTestNames[ii]

            testRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" not in f"{item}") and ("thin" not in f"{item}".lower()))]
            trueRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" in f"{item}")     and ("thin" not in f"{item}".lower()))]

            ### For every episode ###
            for episodePath in testRecord:
                print( f"\n{episodePath}, {int(os.path.getsize(episodePath)/1e6)}MB" )
                try:
                    reader = EROM_Reader( episodePath, suppressLoad = True )
                except RuntimeError:
                    print( f"\nSKIPPED: {episodePath}\n" )
                    continue

                # reader.action_failure_vs_position_variation()
                resBad, resStp = reader.planning_failure_vs_hallucination()
                totalBad += resBad
                totalSteps += resStp
                
    print( f"\n\n{totalBad}/{totalSteps} = {totalBad/totalSteps*100.0}% BAD PLANNING ATTEMPTS\n\n" )
                

                # if _THINIFY:
                #     reader.thinify_recordings_as_files()
                #     reader.erase() # Flush main recording from memory

                #     epPrefix = f"{episodePath}".replace( ".pkl", "" )

                #     statePaths = [item for item in trueRecord if ((epPrefix in f"{item}") and ("_OCV-State" in f"{item}") and (os.path.getsize(item) >= _MIN_STATE_SIZE_BYTES))    ]
                #     statePaths.sort( key = lambda x: dex_key( x ) )

                #     for _j_, sPath in enumerate( statePaths ):
                #         print_header( f"STATE {_j_ + 1}, {sPath}", preWidth = 5, totWidth = 75, capitalize = False )

                #         reader.thinify_state_file( sPath )

except KeyboardInterrupt:
    print( "\nSESSION ENDED BY USER!\n" )

########## EXIT ####################################################################################
crash_out( notify = False )