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

from analysis_Utils import Loc, print_header, dex_key, crash_out
from Example import Example

##### Environment && Constants ############################################
set_blocks_env()
set_experiment_env()
set_render_env()

##### MAIN

try:

    examples : Deque[Example] = deque()

    ### For every block set ###
    for iii, paths in enumerate( Loc.datasets ):

        setNam = Loc.dataLabels[iii]

        ### For every scenario ###
        for ii, test in enumerate( Loc.tests ):

            print_header( f"TEST, {setNam}: {test}", preWidth = 10, totWidth = 100, capitalize = True )

            ##### Init ####################################################
            path     = paths[ii]
            longTNam = Loc.longTestNames[ii]

            testRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" not in f"{item}") and ("thin" not in f"{item}".lower()))]
            trueRecord = [os.path.join( path, item ) for item in sorted( os.listdir( path ) ) if ((".pkl" in f"{item}".lower()) and ("_OCV-State" in f"{item}")     and ("thin" not in f"{item}".lower()))]

            ### For every episode ###
            for _i_, episodePath in enumerate( testRecord ):
                print( f"\n{episodePath}, {int(os.path.getsize(episodePath)/1e6)}MB" )
                try:
                    reader = EROM_Reader( episodePath, suppressLoad = False )
                except RuntimeError:
                    print( f"\nSKIPPED: {episodePath}\n" )
                    continue

                epPrefix = f"{episodePath}".replace( ".pkl", "" )

                statePaths = [item for item in trueRecord if ((epPrefix in f"{item}") and ("_OCV-State" in f"{item}") and ("THIN" not in f"{item}") and (os.path.getsize(item) >= Loc._MIN_STATE_SIZE_BYTES))    ]
                statePaths.sort( key = lambda x: dex_key( x ) )

                ### For every State ###
                for _j_, sPath in enumerate( statePaths ):

                    print_header( f"STATE {_j_ + 1}, {sPath}", preWidth = 5, totWidth = 75, capitalize = False )

                    ex_ij = Example()
                    
                    ##### Address ################
                    ex_ij.dataset = setNam
                    ex_ij.test    = test
                    ex_ij.episode = _i_
                    ex_ij.step    = _j_

                    ##### Perception #############
                    ##### Planning ###############
                    ##### Action #################
                    ##### Next State #############



                    

except KeyboardInterrupt:
    print( "\nPROCESSING ENDED BY USER\n\n" )


########## EXIT ####################################################################################
crash_out( notify = False )