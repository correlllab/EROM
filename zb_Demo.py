########## INIT ####################################################################################

import numpy as np
from py_trees.composites import Sequence

from aspire.symbols import GraspObj, extract_pose_as_homog
from aspire.env_config import env_var


########## EMERGENCY DEMO ENGINEERING ##############################################################


class Block_Pusher( Sequence ):
    """ Attempt to push a block in a target direction """

    def __init__( self, symbol, vector, target, zSAFE = 0.150, name = "Rearrange", ctrl = None ):
        """Construct the subtree"""
        super().__init__( name = name, memory = 1 )
        self.ctrl = ctrl

        backLen = env_var("_BLOCK_SCALE") * 2.0
        symPosn = extract_pose_as_homog( symbol )
        endPosn = np.add( symPosn, vector )

        # FIXME, START HERE: COMPUTE THE MOVES TO PUSH THE BLOCK

        self.add_children([

        ])
        
        # 1. Open the gripper
        if preGraspW_m is None:
            self.add_child(  Open_Gripper( name = "Open", ctrl = ctrl )  )
        else:
            self.add_child(  Set_Gripper( preGraspW_m, name = "Open", ctrl = ctrl )  )
        # 2. Jog to the target
        self.add_child(  Jog_Safe( target, zSAFE = zSAFE, name = "Jog to Grasp Pose", ctrl = ctrl )  )
        # 1. Close the gripper
        if graspWdth_m is None:
            self.add_child(  Close_Gripper( name = "Close", ctrl = ctrl )  )
        else:
            self.add_child(  Set_Gripper( graspWdth_m, name = "Close", ctrl = ctrl )  )



def get_bt_scene_rearranger( symbols : list[GraspObj] ):
    pass
