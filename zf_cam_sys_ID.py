########## INIT ####################################################################################

##### Imports #####

### Standard ###
import pickle, os, sys

### Local ###
from aspire.env_config import set_camera_env, set_object_env
from TS.MinCtrl import MinController, _SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4




########## MAIN ####################################################################################
_GET_DATA = False
_VIZ_DATA = True
_DATA_FIL = "data/CameraTSData.pkl"

if __name__ == "__main__":

    ##### Collect Data ####################################################

    if _GET_DATA:
        
        try:
            set_object_env()
            set_camera_env()

            ctrl = MinController()
            ctrl.start()

            data = ctrl.perceive_at_poses( [_SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4,] )

            with open( _DATA_FIL, 'wb' ) as f:
                pickle.dump( data, f )

            # FIXME: RENDER PCDs!

            ctrl.shutdown()

        except KeyboardInterrupt:
            ctrl.shutdown()

    
    
    ##### Visualize Data ##################################################

    if _VIZ_DATA:
        with open( _DATA_FIL, 'rb' ) as f:
            data = pickle.load( f )
            print( f"There are {len(data)} data elements!" )
            for datum in data:
                # print( list( datum.keys() ), sys.getsizeof( datum ) ) # ['seq', 'pose', 'obrv', 'meta']
                obsrv = datum['obrv']
                for obs in obsrv:
                    # print( type( obs ) )
                    # print( list(obs.keys()) ) # ['Score', 'Probability', 'Count', 'bbox', 'Pose', 'Time', 'CPCD', 'shotID', 'camRay']
                    cpcd = obs['CPCD']
                    # print( type( cpcd ) )
                    # print( list(cpcd.keys()), type( cpcd['points'] ), type( cpcd['colors'] ) ) # np.ndarray


    ##### CRASH OUT #######################################################
    os.system( 'kill %d' % os.getpid() ) 