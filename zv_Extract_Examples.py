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
from Reader import EROM_Reader, ConfMatx

from homog_utils import homog_xform, diff_mag

from analysis_Utils import Loc, print_header, dex_key, crash_out
from Example import Example


##### Environment && Constants ############################################
set_blocks_env()
set_experiment_env()
set_render_env()



########## MAIN ####################################################################################

def get_confusion( matches ):
    rtnLst = deque()
    for match in matches:
        if match[0] != match[1]:
            rtnLst.append( match )
    return list( rtnLst )


try:

    examples : Deque[Example] = deque()
    _EXAMPL_PATH = f"{Loc._MISC_DIR}AllExamples.pkl" 

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