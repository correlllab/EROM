""" ########## DEV_PLAN ##########
[ ] Move robot to 4 poses
    [ ] Collect data
[ ] Iterate transform at each pose until the centers of the objects align
    * I think we basically guessed it an this always bugged me
    [ ] Particle Swarm Optimization
        [ ] Begin with translation only
"""

########## INIT ####################################################################################
import os
from time import sleep
from pprint import pprint
from random import random
from collections import deque
from uuid import uuid4

import numpy as np

from magpie_control.ur5 import UR5_Interface, _CAMERA_XFORM
from magpie_control.poses import translation_diff
from aspire.env_config import set_camera_env, set_object_env
from aspire.utils import normalize_dist
from OWLv2_Segment import Perception_OWLv2, _QUERIES
from utils import zip_dict_sorted_by_decreasing_value




########## CONTROLLER ##############################################################################

##### Params #####
_MOVE_SLEEP_S = 0.5
_LIN_SPEED    = 0.25
_LIN_ACCEL    = 0.5
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

_POSE_4 = np.array( [[ 0.5943,  0.6201,  0.5122, -0.4646],
                     [ 0.5323, -0.7806,  0.3276, -0.4758],
                     [ 0.6029,  0.0779, -0.794 ,  0.3598],
                     [ 0.    ,  0.    ,  0.    ,  1.    ],] )


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
                'pose': self.robot.get_tcp_pose(), # Use actual instead of ideal
                'obrv': obsrv,
                'meta': metadata,
            })
        return self.data



########## PARTICLE SWARM OPTIMIZATION #############################################################

def randrange_f( lo, hi ) -> float:
    """ Return a random number within a float range """
    span = hi - lo
    return lo + span * random()


def sample_bbox( bbox ) -> np.ndarray:
    """ Generate a uniformly random coordinate within a `bbox` """
    rtnPnt = list()
    for coordRange in bbox:
        rtnPnt.append( randrange_f( coordRange[0], coordRange[1] ) )
    return np.array( rtnPnt )


def vector_noise_by_dim( stepArr ) -> np.ndarray:
    """ Generate a random vec with per-dim magnitude `stepArr` """
    return sample_bbox( [[-step,step] for step in stepArr] )


class Particle:
    """ Element of PSO problem """

    def __init__( self, coords : np.ndarray ):
        """ Create a particle """
        # Current
        self.xCurr = coords.copy()
        self.score = -1e9
        # Best
        self.xBest = coords.copy()
        self.sBest = -1e9
        # Delta
        self.veloc = np.zeros( coords.shape )



class CameraPSO:
    """ Iterate on collected data to arrive at a new camera transform """
    # NOTE: There was probably a reasoned way to do this, but I'm just gonna fuzz it

    def __init__( self, bbox, data, Nprt = 1000 ):
        """ Setup problem """
        self.N        = Nprt
        self.bbox     = bbox
        self.prtcls   = deque()
        self.data     = data
        self.Lambda   = 0.75
        self.phiGlob  = (1.0-self.Lambda)/2.0
        self.phiBest  = (1.0-self.Lambda)/2.0
        # self.randStep = np.array( [0.0125, 0.0125, 0.0125,] )
        self.randStep = np.array( [0.005, 0.005, 0.005,] )
        self.pBest    = None


    def swarm_init( self ):
        """ Generate initial points uniformly """
        self.prtcls = deque()
        for _ in range( self.N ):
            part = Particle( sample_bbox( self.bbox ) )
            part.veloc = vector_noise_by_dim( [0.010, 0.010, 0.010,] )
            self.prtcls.append( part )
        self.pBest = self.prtcls[0]


    def particle_2_xform( self, prtcl : Particle ):
        """ Convert to the thing we are optimizing """
        xform = _CAMERA_XFORM.copy()
        xform[0:3,3] = prtcl.xCurr
        return xform

    
    def eval_particle( self, prtcl : Particle ):
        """ Eval the disparity between the measured poses """
        xform  = self.particle_2_xform( prtcl )
        blocks = dict()
        for datum in self.data:
            robotPose_i = datum['pose']
            observtns_i = datum['obrv']
            for obsrv in observtns_i:
                dist = zip_dict_sorted_by_decreasing_value( normalize_dist( obsrv['Probability'] ) )
                labl = dist[0][0]
                prob = dist[0][1]
                pose = robotPose_i.dot( xform ).dot(  np.array( obsrv['Pose'] ).reshape( (4,4,) )  ) 
                item = (pose, prob,)
                if labl not in blocks:
                    blocks[ labl ] = [ item, ]
                else:
                    blocks[ labl ].append( item )
        totScore = 0.0
        for poseList in blocks.values():
            for pair_i in poseList:
                for pair_j in poseList:
                    # totScore -= translation_diff( pair_i[0], pair_j[0] ) * max( pair_i[1], pair_j[1] )
                    totScore -= translation_diff( pair_i[0], pair_j[0] )
        prtcl.score = totScore
        if totScore > prtcl.sBest:
            prtcl.sBest = totScore
            prtcl.xBest = prtcl.xCurr.copy()
        if totScore > self.pBest.score:
            self.pBest = Particle( prtcl.xCurr )
            self.pBest.score = prtcl.score
        

    def eval_swarm( self ):
        """ Score the current positions of all particles """
        # Calc scores
        totSwarm = 0.0
        for prtcl in self.prtcls:
            self.eval_particle( prtcl )
            totSwarm += prtcl.score
        print( f"Average Score: {totSwarm / self.N}" )
        
        # Calc velocities
        totSwarm = 0.0
        for prtcl in self.prtcls:
            prtcl.veloc = prtcl.veloc*self.Lambda + (prtcl.xBest - prtcl.xCurr)*self.phiBest + (self.pBest.xCurr - prtcl.xCurr)*self.phiGlob
            totSwarm += np.linalg.norm( prtcl.veloc )
        print( f"Average Veloc: {totSwarm / self.N}" )


    def update_swarm( self ):
        """ Update the current positions of all particles """
        for prtcl in self.prtcls:
            prtcl.xCurr = prtcl.xCurr + (prtcl.veloc + vector_noise_by_dim( self.randStep ) )


    def run_N_iter( self, Nrun ):
        """ Run `Nrun` iterations of PSO, Return the best transform """
        self.swarm_init()
        for _ in range( Nrun ):
            self.eval_swarm()
            self.update_swarm()
            for prtcl in self.prtcls:
                prtcl.xCurr += vector_noise_by_dim( self.randStep * 1.0 )
                # prtcl.xCurr += vector_noise_by_dim( self.randStep * 2.0 )
                # prtcl.xCurr += vector_noise_by_dim( self.randStep * 5.0 )
                prtcl.veloc += vector_noise_by_dim( self.randStep / 2.0 )
                # prtcl.veloc += vector_noise_by_dim( self.randStep / 5.0 )
            print( f"Best: {self.pBest.xCurr}, {self.pBest.score}" )
        return self.particle_2_xform( self.pBest )







########## MAIN ####################################################################################
if __name__ == "__main__":

    _MEAS_ONLY = False

    try:
        set_object_env()
        set_camera_env()
        
        ctrl = CameraInspector( noViz = _MEAS_ONLY )
        ctrl.start()

        if not _MEAS_ONLY:
            data = ctrl.perceive_at_poses( [_SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4,] )
            # pprint( data )

            pso      = CameraPSO( [[-0.075,+0.075],[-0.075,+0.075],[-0.200,+0.200],], data, Nprt = 2000 )
            camXform = pso.run_N_iter( 200 )

            print( f"Winning Camera Transform:\n{camXform}" )


        ctrl.shutdown()

    except KeyboardInterrupt:
        ctrl.shutdown()

    # CRASH OUT
    os.system( 'kill %d' % os.getpid() ) 
