########## INIT ####################################################################################

##### Imports #####

### Standard ###
import pickle, os

### Local ###
from aspire.env_config import set_camera_env, set_object_env
from TS.MinCtrl import MinController, _SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4




########## MAIN ####################################################################################
if __name__ == "__main__":

    try:
        set_object_env()
        set_camera_env()

        ctrl = MinController()
        ctrl.start()

        data = ctrl.perceive_at_poses( [_SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4,] )

        with open( "data/CameraTSData.pkl", 'wb' ) as f:
            pickle.dump( data, f )

        # FIXME: RENDER PCDs!

        ctrl.shutdown()

    except KeyboardInterrupt:
        ctrl.shutdown()

    # CRASH OUT
    os.system( 'kill %d' % os.getpid() ) 
    