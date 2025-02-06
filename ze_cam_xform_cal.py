""" ########## DEV_PLAN ##########
[ ] Move robot to 4 poses
    [ ] Collect data
[ ] Iterate transform at each pose until the centers of the objects align
    * I think we basically guessed it an this always bugged me
    [ ] Particle swarm optimization
"""

########## INIT ####################################################################################
import os
from time import sleep
from pprint import pprint

import numpy as np

from magpie_control.ur5 import UR5_Interface
from OWLv2_Segment import Perception_OWLv2, _QUERIES



########## CONTROLLER ##############################################################################

##### Params #####
_MOVE_SLEEP_S = 0.5
_LIN_SPEED    = 0.25
_LIN_ACCEL    = 0.5
_MEAS_ONLY    = True
_SAFE_POSE    = np.array( [[-0.996,  0.004, -0.093, -0.178],
                           [ 0.01 ,  0.998, -0.065, -0.278],
                           [ 0.093, -0.066, -0.994,  0.5  ],
                           [ 0.   ,  0.   ,  0.   ,  1.   ],] )

##### Poses #####
_POSE_1 = np.array( [[-0.747, -0.411, -0.522, -0.006],
                     [-0.317,  0.911, -0.264, -0.222],
                     [ 0.584, -0.032, -0.811,  0.343],
                     [ 0.   ,  0.   ,  0.   ,  1.   ],] )

_POSE_2 = np.array( [[-0.695,  0.505, -0.512, -0.038],
                     [ 0.322,  0.855,  0.407, -0.513],
                     [ 0.643,  0.118, -0.757,  0.338],
                     [ 0.   ,  0.   ,  0.   ,  1.   ],] )

_POSE_3 = np.array( [[-0.63 ,  0.438,  0.641, -0.56 ],
                     [ 0.297,  0.899, -0.323, -0.139],
                     [-0.718, -0.013, -0.696,  0.389],
                     [ 0.   ,  0.   ,  0.   ,  1.   ],] )

_POSE_4 = np.array( [[-0.63 ,  0.438,  0.641, -0.56 ],
                     [ 0.297,  0.899, -0.323, -0.139],
                     [-0.718, -0.013, -0.696,  0.389],
                     [ 0.   ,  0.   ,  0.   ,  1.   ],] )


class CameraInspector:
    """ Minimal controller to investigate camera transform """

    def __init__( self, noViz = False ):
        """ Load control and perception interfaces """
        self.noViz = noViz
        if self.noViz:
            self.perc = None
        else:
            self.perc = Perception_OWLv2()
        self.robot : UR5_Interface = UR5_Interface()
        self.data     = list()
        self.safePose = None


    def start( self ):
        """ Start control and perception interfaces """
        self.robot.start()
        if not self.noViz:
            self.perc.start_vision()
        else:
            sleep( 2.0 )
            print( self.robot.get_tcp_pose() )


    def shutdown( self ):
        """ Stop the Perception Process and the UR5 connection """
        self.robot.reset_gripper_overload( restart = False )
        self.robot.stop()
        if not self.noViz:
            self.perc.shutdown()


    def move_arm_to_pose( self, poseHomog ):
        """ Move the arm linearly thru task space, then brief pause """
        self.robot.moveL( poseHomog, _LIN_SPEED, _LIN_ACCEL, False )
        sleep( _MOVE_SLEEP_S )


    def perceive_at_poses( self, poseList : list[np.ndarray] ):
        """ Go to each pose, then run the perception pipeline """
        self.data = list() # Erase data
        for i, pose in enumerate( poseList ):
            self.move_arm_to_pose( _SAFE_POSE )
            self.move_arm_to_pose( pose )
            obsrv, metadata = self.perc.segment( _QUERIES )
            self.data.append({
                'seq' : i,
                'pose': np.array( pose ),
                'obrv': obsrv,
                'meta': metadata,
            })
        return self.data



########## MAIN ####################################################################################
if __name__ == "__main__":

    try:
        ctrl = CameraInspector( noViz = _MEAS_ONLY )
        ctrl.start()

        if not _MEAS_ONLY:
            data = ctrl.perceive_at_poses( [_SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4,] )
            pprint( data )

        ctrl.shutdown()

    except KeyboardInterrupt:
        ctrl.shutdown()

    # CRASH OUT
    os.system( 'kill %d' % os.getpid() ) 
