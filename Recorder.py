########## INIT ####################################################################################

import os
from time import sleep

import numpy as np

from magpie_control.ur5 import UR5_Interface

### ASPIRE ###
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import _SHOT_1, _SHOT_2, _SHOT_3, _SHOT_4, _SHOT_5, _SHOT_6, set_experiment_env
from Memory import Memory
from obj_ID_server import Perception_OWLViT
from OWLv2_Segment import Perception_OWLv2, _QUERIES



########## SETTINGS ################################################################################
_USE_OWL_2 = True



def record_readings( shotList : list[np.ndarray], N : int ):
    """ Characterize the vision system """
    set_blocks_env()
    set_experiment_env()

    if _USE_OWL_2:
        prc = Perception_OWLv2
    else:
        prc = Perception_OWLViT

    mem = Memory()
    rbt = UR5_Interface()

    prc.start_vision()
    rbt.start()
    rbt.set_grip_N( 10.0 )

    for i in range(N):
        for j, shotPose in enumerate( shotList ):
            mem.history.append( msg = "step", datum = [i,j,] )
            rbt.moveL( shotPose, asynch = False )
            sleep( 2.0 )
            camPose = rbt.get_cam_pose()
            mem.history.append( msg = "camera", datum = camPose.copy() )

            if _USE_OWL_2:
                obsrv = prc.segment( _QUERIES )
            else:
                obsrv = prc.build_model( shots = 1 )
            
            mem.process_observations( 
                obsrv,
                camPose 
            ) 
            mem.get_current_most_likely()
    
    mem.shutdown()
    prc.shutdown()
    rbt.stop()


if __name__ == "__main__":
    try:
        record_readings( [_SHOT_1, _SHOT_2, _SHOT_3, _SHOT_4, _SHOT_5, _SHOT_6,], 4 )
    except Exception as e:
        print( f"BAD THING: {e}" )
    except KeyboardInterrupt:
        print( "\nSTOPPED by user!\n" )

    os.system( f'kill {os.getpid()}' ) 
