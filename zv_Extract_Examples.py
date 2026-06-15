########## INIT ####################################################################################
import pickle, os, gc, traceback, json

from collections import deque
from typing import Any, Deque
from pprint import pprint
from random import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from aspire.BlocksTask import set_blocks_env
from aspire.env_config import env_var
from aspire.symbols import GraspObj, extract_position
from TaskPlanner import set_experiment_env
from draw_beliefs import set_render_env
from magpie_control.realsense_wrapper import MPCD
from Reader import EROM_Reader, ConfMatx

from homog_utils import homog_xform, diff_mag

from analysis_Utils import Loc, print_header, dex_key, crash_out
from Example import Example, ActionStatus


##### Environment && Constants ############################################
set_blocks_env()
set_experiment_env()
set_render_env()



########## HELPER CLASSES ##########################################################################

class Example_Reader:
    """ Treat the examples like a database """
    def __init__( self, path : str = None ):
        """  """
        if path is None:
            self.path = _EXAMPL_PATH
        else:
            self.path = path
        self.data : deque[Example] = None
        self.slct : deque[Example] = deque()
        try:
            with open( self.path, 'rb' ) as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            print( f"There was NO pickle file at {self.path}" )


    def select( self, dataset, test, episode = None, step = None ):
        """ Set the selection and return a reference to it """
        self.slct = deque()
        for datum in self.data:
            if datum.dataset == dataset and datum.test == test:
                if episode is None:
                    self.slct.append( datum )
                elif datum.episode == episode:
                    if step is None:
                        self.slct.append( datum )
                    elif datum.step == step:
                        self.slct.append( datum )
        return self.slct


    @staticmethod
    def num_blocks_in_column( blocks : dict[str,GraspObj] ):
        xyTrgt = np.array( [ -0.200, -0.300,] )
        zUnit  = env_var("_BLOCK_SCALE")
        zHalf  = zUnit / 2.0
        blcOK  = 0
        factor = 0.80
        for i in range(3):
            found = False
            for val in blocks.values():
                posn_j = extract_position( val )
                xyOK   = np.linalg.norm( xyTrgt - posn_j[:2] ) <= env_var("_PLACE_XY_ACCEPT")
                zOK    = np.abs( (zHalf + i * zUnit) - posn_j[2] ) <= zHalf * factor
                if xyOK and zOK:
                    found = True
                    break
            if found:
                blcOK += 1
        return blcOK

        
    def quantify_steps_lost_on_action_failure( self ):
        """ Get metric for the selection """
        lostDque = deque()
        ## For each example in the selection ##
        for datum in self.slct:
            if datum.nextState is None:
                continue
            ## If an action failure occurred this step ##
            if datum.action == ActionStatus.FAILURE:
                ## Determine how many steps were lost ##
                nBefor = Example_Reader.num_blocks_in_column( datum.trueSymbols )
                nAfter = Example_Reader.num_blocks_in_column( datum.nextState.trueSymbols )
                lostDque.append( nBefor - nAfter )
        return list( lostDque )
                



########## MAIN ####################################################################################
_EXAMPL_PATH = f"{Loc._MISC_DIR}AllExamples.pkl" 
_AGGREGATE   = False
_GET_METRICS = True

def get_confusion( matches ):
    rtnLst = deque()
    for match in matches:
        if match[0] != match[1]:
            rtnLst.append( match )
    return list( rtnLst )


"""
########## DISTRIBUTION OF STEPS LOST ON ACTION FAILURE ############################################

##### TEST, RGB: KC-KP ####################################################
{}

##### TEST, RGB: SC-KP ####################################################
{-1: 0.09375, 0: 0.625, 1: 0.25, 2: 0.03125}

##### TEST, RGB: KC-SP ####################################################
{-1: 0.030303030303030304,
 0: 0.7272727272727273,
 1: 0.21212121212121213,
 2: 0.030303030303030304}

##### TEST, RGB: SC-SP ####################################################
{-1: 0.06493506493506493,
 0: 0.8051948051948052,
 1: 0.11688311688311688,
 2: 0.012987012987012988}

##### TEST, RBW: KC-KP ####################################################
{1: 1.0}

##### TEST, RBW: SC-KP ####################################################
{-1: 0.05714285714285714,
 0: 0.8,
 1: 0.11428571428571428,
 2: 0.02857142857142857}

##### TEST, RBW: KC-SP ####################################################
{-1: 0.047619047619047616, 0: 0.7619047619047619, 1: 0.19047619047619047}

##### TEST, RBW: SC-SP ####################################################
{-1: 0.08163265306122448,
 0: 0.673469387755102,
 1: 0.20408163265306123,
 2: 0.04081632653061224}

"""


if _GET_METRICS:
    try:

        exReader = Example_Reader( _EXAMPL_PATH )

        ### For every block set ###
        for iii, paths in enumerate( Loc.datasets ):

            setNam   = Loc.dataLabels[iii]
            classes  = Loc.blcNam[setNam]
            eClasses = Loc.eBlcNam[setNam]

            ### For every scenario ###
            for ii, test in enumerate( Loc.tests ):

                print_header( f"TEST, {setNam}: {test}", preWidth = 10, totWidth = 100, capitalize = True )
                
                exReader.select( setNam, test )
                lostStep = exReader.quantify_steps_lost_on_action_failure()
                keys     = set( lostStep )
                N        = len( lostStep )
                dist     = dict()

                for k in keys:
                    dist[k] = 0

                for loss_i in lostStep:
                    dist[ loss_i ] += 1

                for k in keys:
                    dist[k] /= N

                pprint( dist )
                print()


    except KeyboardInterrupt:
        print( "\nPROCESSING ENDED BY USER\n\n" )


if _AGGREGATE:
    try:

        examples : Deque[Example] = deque()

        ### For every block set ###
        for iii, paths in enumerate( Loc.datasets ):

            setNam   = Loc.dataLabels[iii]
            classes  = Loc.blcNam[setNam]
            eClasses = Loc.eBlcNam[setNam]

            ### For every scenario ###
            for ii, test in enumerate( Loc.tests ):

                confMatx = ConfMatx()
                confMatx.add_classes( eClasses )

                print_header( f"TEST, {setNam}: {test}", preWidth = 10, totWidth = 100, capitalize = True )

                ##### Init ####################################################
                path     = paths[ii]
                longTNam = Loc.longTestNames[ii]

                testRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" not in f"{item}") and ("thin" not in f"{item}".lower()))]
                trueRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" in f"{item}")     and ("thin" not in f"{item}".lower()))]

                ### For every episode ###
                for _i_, episodePath in enumerate( testRecord ):
                    print( f"\n{episodePath}, {int(os.path.getsize(episodePath)/1e6)}MB" )
                    reader : EROM_Reader = None
                    try:
                        reader = EROM_Reader( episodePath, suppressLoad = True )
                    except RuntimeError:
                        print( f"\nSKIPPED: {episodePath}\n" )
                        continue

                    epPrefix = f"{episodePath}".replace( ".pkl", "" )

                    statePaths = [item for item in trueRecord if ((epPrefix in f"{item}") and ("_OCV-State" in f"{item}") and ("THIN" not in f"{item}") and (os.path.getsize(item) >= Loc._MIN_STATE_SIZE_BYTES))    ]
                    statePaths.sort( key = lambda x: dex_key( x ) )
            
                    assocStates, assocSteps = reader.get_states_and_steps()

                    for _j_ in range( len( assocStates ) ):
                        fState_j = assocStates[_j_]
                        fStep_j  = assocSteps[_j_]

                        with open( fState_j, 'rb' ) as f:
                            state_j = pickle.load(f)
                        with open( fStep_j, 'rb' ) as f:
                            step_j = pickle.load(f)

                        print_header( f"STATE {_j_ + 1}, {fState_j}", preWidth = 5, totWidth = 75, capitalize = False )

                        ex_ij = Example()
                        
                        ##### State ##################

                        ex_ij.sensBeliefs  = state_j["sensBeliefs"]
                        ex_ij.trueReadings = state_j["trueReadings"]
                        ex_ij.sensSymbols  = state_j["sensSymbols"]
                        ex_ij.trueSymbols  = state_j["trueSymbols"]

                        ##### Address ################
                        ex_ij.dataset = setNam
                        ex_ij.test    = test
                        ex_ij.episode = _i_
                        ex_ij.step    = _j_

                        ##### Perception #############
                        # state_j.keys() # 'labels', 'image', 'depth', 'clouds', 'sensBeliefs', 'trueReadings', 'sensSymbols', 'trueSymbols'
                        ex_ij.p_AllBlocks = len( state_j["sensSymbols"] ) >= 3 
                        
                        reqSet = set( classes )
                        fndSet = set( [item.label for item in state_j["sensSymbols"]] )
                        ex_ij.missingBlock = list( reqSet.difference( fndSet ) )

                        matches = confMatx.match( list( state_j["trueSymbols"].values() ), state_j["sensSymbols"] )
                        confusn = get_confusion( matches )
                        ex_ij.p_Confused   = (len( confusn ) > 0)
                        ex_ij.confusedBloc = confusn

                        ex_ij.positionErrs = EROM_Reader.position_error_from_state( state_j )

                        ex_ij.N_halluc = max( 0, len( state_j["sensSymbols"] )-3 )

                        ##### Planning ###############
                        ex_ij.planned = reader.planning_result_from_thin_step( step_j, state_j )

                        ##### Action #################
                        ex_ij.action = reader.action_result_from_thin_step( step_j )

                        ##### Next State #############
                        if _j_ > 0:
                            examples[-1].nextState = ex_ij

                        # pprint( ex_ij )
                        examples.append( ex_ij )

            #             break
            #         break
            #     break
            # break
        
        with open( _EXAMPL_PATH, 'wb' ) as f:
            pickle.dump( examples, f )

    except KeyboardInterrupt:
        print( "\nPROCESSING ENDED BY USER\n\n" )


########## EXIT ####################################################################################
crash_out( notify = False )