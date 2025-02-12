########## INIT ####################################################################################

##### Imports #####

### Standard ###
import pickle

### Local ###
from TS.MinCtrl import MinController, _SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4



########## MAIN ####################################################################################
if __name__ == "__main__":

    ctrl = MinController()
    ctrl.start()

    data = ctrl.perceive_at_poses( [_SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4,] )

    with open( "CameraTSData.pkl", 'wb' ) as f:
        pickle.dump( data, f )

    # FIXME: RENDER PCDs!