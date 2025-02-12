########## INIT ####################################################################################

##### Imports #####

### Standard ###
import pickle, os, sys

### Special ###
import numpy as np

### MAGPIE ###
from magpie_control.ur5 import _CAMERA_XFORM

### ASPIRE ###
from aspire.env_config import set_camera_env, set_object_env
from aspire.BlocksTask import set_blocks_env
from aspire.symbols import CPCD

### Local ###
from TaskPlanner import set_experiment_env
from TS.MinCtrl import MinController, _SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4
from draw_beliefs import points_colors_geo, vispy_geo_list_window, set_render_env, cpcd_geo




########## MAIN ####################################################################################
_GET_DATA = False
_VIZ_DATA = True
_DATA_FIL = "data/CameraTSData.pkl"
_TRANSFRM = True

if __name__ == "__main__":

    poseSeq = [_SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4,]

    ##### Collect Data ####################################################

    if _GET_DATA:
        
        try:
            set_object_env()
            set_camera_env()

            ctrl = MinController()
            ctrl.start()

            data = ctrl.perceive_at_poses( poseSeq )

            with open( _DATA_FIL, 'wb' ) as f:
                pickle.dump( data, f )

            ctrl.shutdown()

        except KeyboardInterrupt:
            ctrl.shutdown()

    
    
    ##### Visualize Data ##################################################

    if _VIZ_DATA:
        set_blocks_env()
        set_experiment_env()
        set_render_env()

        geoLst = list()
        with open( _DATA_FIL, 'rb' ) as f:
            data = pickle.load( f )
            print( f"There are {len(data)} data elements!" )
            for i, datum in enumerate( data ):
                rbtPos = poseSeq[i]
                obsrv  = datum['obrv']
                for obs in obsrv:
                    # print( type( obs ) )
                    # print( list(obs.keys()) ) # ['Score', 'Probability', 'Count', 'bbox', 'Pose', 'Time', 'CPCD', 'shotID', 'camRay']
                    cpcd = obs['CPCD']
                    # print( type( cpcd ) )
                    # print( list(cpcd.keys()), type( cpcd['points'] ), type( cpcd['colors'] ) ) # np.ndarray
                    if not _TRANSFRM:
                        geoLst.extend( points_colors_geo( cpcd['points'], cpcd['colors'] ) )
                    else:
                        obj   = CPCD( points = cpcd['points'], colors = cpcd['colors'] )
                        xform = rbtPos.dot( _CAMERA_XFORM )
                        print( xform )
                        obj.transform( xform )
                        geoLst.extend( cpcd_geo( obj ) )
        vispy_geo_list_window( geoLst )


    ##### CRASH OUT #######################################################
    os.system( 'kill %d' % os.getpid() ) 