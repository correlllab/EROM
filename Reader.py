########## INIT ####################################################################################
import pickle, os, gc, traceback, json

from collections import deque
from typing import Any, Deque
from pprint import pprint
from random import random

import matplotlib.pyplot as plt
import numpy as np

from aspire.symbols import GraspObj, ObjPose, extract_position, euclidean_distance_between_symbols
from aspire.BlocksTask import set_blocks_env
from aspire.env_config import env_var
from TaskPlanner import set_experiment_env
from draw_beliefs import set_render_env
from magpie_control.realsense_wrapper import MPCD

from homog_utils import homog_xform, diff_mag, posn_from_xform
from utils import dex_key, parse_action
from Example import PlanStatus, ActionStatus

set_blocks_env()

# _BLOCK_DIAG = np.nan
# _HALF_SCALE = np.nan

# _BLOCK_DIAG = np.sqrt( 3 * env_var("_BLOCK_SCALE")**2 )
# _HALF_SCALE = env_var("_BLOCK_SCALE") / 2.0


########## HELPER FUNCTIONS ########################################################################

def pnt_distance_from_line_3D( qX0 : np.ndarray, linX1 : np.ndarray, linX2 : np.ndarray ):
    """ Get the distance of a 3D point `qX0` from an infinite line defined by `linX1` --to-> `linX2` """
    num = np.linalg.norm( np.cross(
        np.subtract( qX0, linX1 ),
        np.subtract( qX0, linX2 )
    ) )
    den = np.linalg.norm( np.subtract( linX2, linX1 ) )
    if den > 0.0:
        return num / den
    else:
        return np.nan



########## CONFUSION MATRIX ########################################################################

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
        
        return assocStates, assocSteps


    @staticmethod
    def parse_plan_from_thin_step( step : list[dict[str,Any]] ):
        """ Return the result of planning """
        for datum in step:
            if "END: Phase 3" in datum["msg"]:
                dtmDat = datum["data"]
                if (dtmDat is not None) and len( dtmDat ):
                    if len( dtmDat["plan"] ):
                        # pprint( dtmDat )
                        return parse_action( dtmDat )
        return None


    @staticmethod
    def p_action_passes_through_known_blocks( state_i : dict[str,list|dict], step_i : list, Zsafe : float = 0.250 ):
        """ Did the planner not correctly ground condtions? """
        # Scale
        _FACTOR     = 0.7 # 0.8
        _HALF_DIAG  = np.sqrt( 3 * env_var("_BLOCK_SCALE")**2 ) / 2.0
        # _HALF_SCALE = env_var("_BLOCK_SCALE") / 2.0
        _D_FUNKY    = _HALF_DIAG * _FACTOR
        # Fetch needed info
        action  = EROM_Reader.parse_plan_from_thin_step( step_i )
        symbols = list( state_i["trueSymbols"].values() )

        if action is None:
            return None

        endPsn = posn_from_xform( action["endPose"] )
        endUpP = endPsn.copy()
        endUpP[2] = Zsafe

        for sym in symbols:
            sPosn = extract_position( sym )
            dStep = pnt_distance_from_line_3D( sPosn, endUpP, endPsn )
            dEndA = np.linalg.norm( sPosn - endPsn )
            # if (dStep < _HALF_SCALE) and (dEndA < _HALF_DIAG):
            if (dStep < _D_FUNKY) and (dEndA < _D_FUNKY):
                return True
        return False
    

    @staticmethod
    def planning_result_from_thin_step( step_i : list[dict[str,Any]], state_i : dict[str,list|dict], prntPlan : bool = False ) -> PlanStatus:
        """ Return the result of planning """
        for datum in step_i:
            try:
                if "END: Phase 3" in datum["msg"]:
                    dtmDat = datum["data"]
                    if (dtmDat is not None) and len( dtmDat ):
                        if len( dtmDat["plan"] ):
                            if prntPlan:
                                print()
                                pprint( dtmDat["next"] )
                                print()
                            if EROM_Reader.p_action_passes_through_known_blocks( state_i, step_i, Zsafe = 0.250 ):
                                return PlanStatus.PLANNED_BAD
                            else:
                                return PlanStatus.PLANNED_OK
                        else:
                            return PlanStatus.NOT_PLANNED
                    elif (dtmDat is None) or (not len( dtmDat )):
                        return PlanStatus.NOT_PLANNED
                    else:
                        raise ValueError( "THIS SHOULD NOT HAPPEN!" )
            except TypeError:
                return PlanStatus.NOT_PLANNED
        if EROM_Reader.p_believe_success_at_thin_step( step_i ):
            return PlanStatus.TASK_DONE
        else:
            return PlanStatus.NOT_PLANNED
    

    @staticmethod
    def p_believe_success_at_thin_step( step : list[dict[str,Any]] ):
        """ Did the system think it succeeded this step? """
        for datum in step:
            dMsg = datum["msg"]
            if "Believe Success" in dMsg:
                return True
        return False


    def count_into_confusion_matrix( self, matx : ConfMatx ):
        assocStates, _ = self.get_states_and_steps()

        for i in range( len( assocStates ) ):
            fState_i = assocStates[i]

            with open( fState_i, 'rb' ) as f:
                state_i = pickle.load(f)

            actl_i = list( EROM_Reader.get_avg_objects( state_i["objects"] ).values() )
            pred_i = state_i["sensSymbols"]
            matx.count_examples( actl_i, pred_i )
    
    
    def planning_failure_vs_hallucination( self ):
        """ What influence do hallucinated blocks have on planning failure? """
        assocStates, assocSteps = self.get_states_and_steps()

        """
        "labels" : list(), #- List of objects in this scene
        "image"  : dict(), #- Lookup of color images used
        "depth"  : dict(), #- Lookup of depth images used
        "clouds" : deque(), # Collection of clouds obtained from the masked images
        "objects": deque(), # Collection of readings obtained from the masked images
        "sensSymbols" : list(), # Collection of symbols obtained from the robot
        "symbols": dict(), #- Lookup of objects obtained from the readings
        """

        totalSteps = len( assocStates )
        totalBad   = 0
        rtnDct     = {
            "N_halluc" : deque(),
            "N_missng" : deque(),
            "resPlan"  : deque(),
        }

        for i in range( len( assocStates ) ):
            fState_i = assocStates[i]
            fStep_i  = assocSteps[i]


            with open( fState_i, 'rb' ) as f:
                state_i = pickle.load(f)
            with open( fStep_i, 'rb' ) as f:
                step_i = pickle.load(f)

            if EROM_Reader.p_believe_success_at_thin_step( step_i ):
                break
            
            # Count hallucinations for this step
            if len( state_i["sensSymbols"] ) > 3:
                rtnDct["N_halluc"].append( len( state_i["sensSymbols"] )-3 )
            else:
                rtnDct["N_halluc"].append( 0 )
            
            # Count missing blocks for this step
            if len( state_i["sensSymbols"] ) < 3:
                rtnDct["N_missng"].append( 3-len( state_i["sensSymbols"] ) )
            else:
                rtnDct["N_missng"].append( 0 )

            rtnDct["resPlan"].append( EROM_Reader.planning_result_from_thin_step( step_i ) )

            if len( state_i["sensSymbols"] ) != 3:
                print()
                if len( state_i["sensSymbols"] ) > 3:
                    print( "HALLUCINATION" )
                elif len( state_i["sensSymbols"] ) < 3:
                    print( "MISSING BLOCK" )
                    if EROM_Reader.planning_result_from_thin_step( step_i ):
                        totalBad += 1

                print( "sensSymbols" )
                for object_j in state_i["sensSymbols"]:
                    print( object_j )

                print( "Symbols" )
                for name, object_j in state_i["symbols"].items():
                    print( name, object_j )

                print( "plan:", EROM_Reader.planning_result_from_thin_step( step_i, prntPlan=True ) )
                print( "actn:", EROM_Reader.action_result_from_thin_step( step_i )   )
            else:
                print( "\nSymbols" )
                for name, object_j in state_i["symbols"].items():
                    print( name, object_j )
            
        print( f"\n\n{totalBad}/{totalSteps} BAD PLANNING ATTEMPTS\n\n" )
        return rtnDct
    

    @staticmethod
    def action_result_from_thin_step( step : list[dict[str,Any]] ) -> ActionStatus:
        """ Return the result of the action """
        for datum in step:
            if "BT END: Status.SUCCESS" in datum["msg"]:
                return ActionStatus.SUCCESS
            if "BT END: Status.FAILURE" in datum["msg"]:
                return ActionStatus.FAILURE
        return ActionStatus.NO_ACTION
    

    @staticmethod
    def get_avg_objects( objLst : list[GraspObj] ) -> dict[str,GraspObj]:
        """ Get the average position from a list of objects """
        rtnObj = dict()
        objDct : dict[str,list[GraspObj]] = dict()
        for obj in objLst:
            if obj.label not in objDct:
                objDct[ obj.label ] = deque()
            objDct[ obj.label ].append( obj )
        
        for lbl, lst in objDct.items():
            avgPsn = np.zeros(3)
            for obj_j in lst:
                avgPsn += extract_position( obj_j )
            avgPsn /= len( lst )
            rtnObj[ lbl ] = GraspObj(
                label = lbl,
                pose  = homog_xform( posnVctr = avgPsn )
            )
        # pprint( rtnObj )
        return rtnObj
    

    @staticmethod
    def position_error_from_state( state_i : dict  ) -> dict:
        """ Get position error for this state """
        truthDict = EROM_Reader.get_avg_objects( state_i["trueReadings"] )
        retrnDict = dict()
        for obj_j in state_i["sensSymbols"]:
            if obj_j.label in truthDict:
                d  = diff_mag(
                    extract_position( obj_j ),
                    extract_position( truthDict[ obj_j.label ] )
                )
                if obj_j.label in retrnDict:
                    if d < retrnDict[ obj_j.label ]:
                        retrnDict[ obj_j.label ] = d
                else: 
                    retrnDict[ obj_j.label ] = d
        return retrnDict


    def action_failure_vs_position_variation( self ):
        """ What influence does Position Variation have on action failure? """
        
        assocStates, assocSteps = self.get_states_and_steps()

        """
        "labels" : list(), #- List of objects in this scene
        "image"  : dict(), #- Lookup of color images used
        "depth"  : dict(), #- Lookup of depth images used
        "clouds" : deque(), # Collection of clouds obtained from the masked images

        "objects": deque(), # Collection of readings obtained from the masked images
        
        "sensSymbols" : list(), # Collection of symbols obtained from the robot
        "symbols": dict(), #- Lookup of objects obtained from the readings
        """

        rtnDct = {
            "avgErr" : deque(),
            "resActn": deque(),
        }

        # 2. Get the position variation of the block to be manipulated
        for i in range( len( assocStates ) ):
            fStep_i  = assocSteps[i]
            fState_i = assocStates[i]
            state_i  = None
            step_i   = None

            with open( fState_i, 'rb' ) as f:
                state_i = pickle.load(f)

            with open( fStep_i, 'rb' ) as f:
                step_i = pickle.load(f)

            if EROM_Reader.p_believe_success_at_thin_step( step_i ):
                break

            stepRes = EROM_Reader.action_result_from_thin_step( step_i )
            rtnDct["resActn"].append( stepRes )
            
            truthDict = EROM_Reader.get_avg_objects( state_i["trueReadings"] )
            
            if len( state_i["sensSymbols"] ):
                N = 0
                d = 0.0
                for lbl, obj_j in state_i["sensSymbols"].items():
                    if lbl in truthDict:
                        N += 1
                        d += diff_mag(
                            extract_position( obj_j ),
                            extract_position( truthDict[ lbl ] )
                        )
                if N:
                    rtnDct["avgErr"].append( d/N )
                else:
                    rtnDct["avgErr"].append( -1.0 ) # Negative means no symbol found
            else:
                rtnDct["avgErr"].append( -1.0 ) # Negative means no symbol found

        return rtnDct