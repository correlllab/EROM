########## INIT ####################################################################################

import numpy as np
from py_trees.composites import Sequence

from magpie_control.poses import vec_unit
from magpie_control.BT import Open_Gripper, Close_Gripper, Jog_Safe, Move_Arm
from magpie_control.utils import vec_diff_mag
from aspire.symbols import GraspObj, extract_pose_as_homog
from aspire.env_config import env_var
from aspire.actions import grasp_pose_from_posn



########## EMERGENCY DEMO ENGINEERING ##############################################################


class Block_Pusher( Sequence ):
    """ Attempt to push a block in a target direction """

    def __init__( self, symbol, vector, zSAFE = 0.150, name = "Rearrange", ctrl = None ):
        """Construct the subtree"""
        super().__init__( name = name, memory = 1 )
        self.ctrl = ctrl

        antiDir = -vec_unit( vector )
        backLen = env_var("_BLOCK_SCALE") * 2.0
        symPosn = extract_pose_as_homog( symbol )[0:3,3]
        bgnPosn = np.add( symPosn, np.multiply( antiDir, backLen ) )
        endPosn = np.add( symPosn, vector )
        bgnPose = grasp_pose_from_posn( bgnPosn )
        endPose = grasp_pose_from_posn( endPosn )
 

        self.add_children([
            Close_Gripper( name = "Close", ctrl = ctrl ),
            Jog_Safe( bgnPose, zSAFE = zSAFE, name = "Jog to Pre-Push", ctrl = ctrl ),
            Move_Arm( endPose, ctrl = ctrl, linSpeed = env_var("_ROBOT_FREE_SPEED")*0.65 ),
        ])
        
        
def get_bt_scene_rearranger( symbols : list[GraspObj], ctrl = None, zSAFE = 0.150 ):
    """ Construct a BT to rearrange the scene """
    nudge    = env_var("_ANGRY_PUSH_M")
    options  = [np.array(item) for item in [[0,nudge,0], [0,-nudge,0],]]
    centroid = np.zeros( (3,) )
    rtnBT    = Sequence( name = "Rearrange Scene", memory = True )
    for sym in symbols:
        centroid += extract_pose_as_homog( sym )[0:3,3]
    centroid /= len(symbols)
    for i, sym in enumerate( symbols ):
        if (( i-1 ) % 2) == 0:
            continue
        sPsn = extract_pose_as_homog( sym )[0:3,3]
        dMax = 0.0
        dVec = np.zeros( (3,) )
        pad  = 0.060
        d    = -1.0
        for optn in options:
            t = sPsn+optn
            d = vec_diff_mag( t, centroid )
            # if not (env_var("_MIN_X_OFFSET") < t[0] < env_var("_MAX_X_OFFSET")):
            #     d = -1.0
            if not ((env_var("_MIN_Y_OFFSET")+pad) < t[1] < (env_var("_MAX_Y_OFFSET")-pad)):
                d = -1.0
            if not ((env_var("_MIN_Y_OFFSET")+pad) < sPsn[1] < (env_var("_MAX_Y_OFFSET")-pad)):
                d = -1.0
            if d > dMax:
                dMax = d
                dVec = optn
        rtnBT.add_child( Block_Pusher( sym, dVec, zSAFE = zSAFE, ctrl = ctrl ) )
    return rtnBT
        
            
        
