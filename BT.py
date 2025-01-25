########## INIT ####################################################################################

##### Imports #####

### Standard ###
from datetime import datetime
from time import sleep

### Special ###
import numpy as np
from py_trees.common import Status
from py_trees.composites import Sequence, Parallel

### Local ###
from magpie_control.BT import BasicBehavior, CycleTimer, SetBBVar, Pause_Robot, Resume_Robot
from magpie_control.ur5 import UR5_Interface
from magpie_control.utils import vec_unit
from aspire.symbols import env_var
from aspire.actions.pdls_behaviors import GroundedAction, MoveFree, MoveFree_w_Pause
from aspire.actions.utils import line_intersect_plane




########## RESPONSIVE PLANNER BEHAVIOR TREES #######################################################


class CheckPerceptionShot( BasicBehavior ):
    """ Ask the Perception Pipeline to get a reading """

    def __init__( self, robot : UR5_Interface = None, name = None ):
        """ Setup """
        super().__init__( name = name, ctrl = robot )


    def update( self ):
        """ Check to see if it's okay to look """

        tcpPose = self.ctrl.get_tcp_pose()
        camPose = np.dot( tcpPose, self.ctrl.camXform )
        zzMag   = camPose[2,2]

        # 1. If downward-facing, then Check for correction
        if (zzMag < 0.0):
            hndZdir = vec_unit( np.dot( camPose, np.array( [0.0, 0.0, -1.0, 1.0,] ) )[0:3] )
            camPosn = camPose[0:3,3]
            XYintrc = line_intersect_plane( camPosn, hndZdir, [0.0, 0.0, 0.0,], [0.0, 0.0, 1.0,], pntParallel = False )
            if XYintrc is None:
                self.status = Status.FAILURE
            else:
                dShot = np.linalg.norm( np.subtract( XYintrc, camPosn ) )
                if dShot >= env_var("_MIN_CAM_PCD_DIST_M"):
                    self.status = Status.SUCCESS
                else:
                    self.status = Status.FAILURE
        else:
            self.status = Status.FAILURE
        return self.status



class PerceiveScene( BasicBehavior ):
    """ Ask the Perception Pipeline to get a reading """

    def __init__( self, robot : UR5_Interface = None, name = None, perceive_cb = None, check_cb = None ):

        self.perc     = perceive_cb
        self.chek     = check_cb
        self.needCool = False
        
        if name is None:
            name = f"PerceiveScene with callbacks {perceive_cb.__name__} and {check_cb.__name__}"
        super().__init__(  name = name, ctrl = robot )


    def initialise( self ):
        """ Actually Move """
        super().initialise()
        if self.ctrl.p_moving():
            timeStr = datetime.now().strftime("%H:%M:%S")
            print( f"\n WARN: `PerceiveScene.initialise`: Robot was MOVING at init time: {timeStr}!\n" )
            self.needCool = True
        else:
            self.needCool = False
    

    def update( self ):
        """ Return true if the target reached """
        if self.needCool:
            sleep( env_var("_MOVE_COOLDOWN_S") )
        if self.ctrl.p_moving():
            timeStr = datetime.now().strftime("%H:%M:%S")
            print( f"\n`PerceiveScene.initialise`: Robot was MOVING at UPDATE time: {timeStr}!\n" )
            self.status = Status.FAILURE
        else:
            self.perc( 1 )
            sleep( 0.25 )
            if self.chek():
                self.status = Status.SUCCESS
            else:
                self.status = Status.FAILURE
        return self.status



########## MOVE AND PERCEIVE #######################################################################

class MoveFree_and_PerceiveScene( GroundedAction ):
    """ Get a replacement sequence for `MoveFree` that stops for perception at the appropriate times """
    # FUTURE: PROBABLY MORE SOPHISTICATED SENSORY PLANNING GOES HERE

    def __init__( self, args, robot = None, name = None, suppressGrasp = False, perceive_cb = None, check_cb = None ):
        """ Init BT """
        self._VERBOSE = True

        # ?poseBgn ?poseEnd
        poseBgn, poseEnd = args
        mfBT = MoveFree_w_Pause( args, robot, name, suppressGrasp )

        root = Parallel( "MoveFree" )
        perc = Sequence( "Stop-and-Perceive", memory = True  )
        perc.add_children([
            CycleTimer( env_var("_UPDATE_PERIOD_S") ),
            CheckPerceptionShot( robot, "Okay for CPCD?" ),
            Pause_Robot(), 
            PerceiveScene( robot, "Check Distribution Change", perceive_cb, check_cb ),
            Resume_Robot(),
        ])
        root.add_children([
            perc,
            mfBT
        ])
        return root
