########## INIT ####################################################################################

##### Imports #####

### Standard ###
import pickle, os, sys

### Special ###
import numpy as np
import cv2
import numpy as np

### MAGPIE ###
from magpie_control.ur5 import _CAMERA_XFORM
from magpie_control.realsense_wrapper import RealSense

### ASPIRE ###
from aspire.env_config import set_camera_env, set_object_env
from aspire.BlocksTask import set_blocks_env
from aspire.symbols import CPCD

### Local ###
from TaskPlanner import set_experiment_env
from TS.MinCtrl import MinController, _SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4
from draw_beliefs import points_colors_geo, vispy_geo_list_window, set_render_env, cpcd_geo



########## HELPER FUNCTIONS ########################################################################

def find_checkerboard_pose( image : np.ndarray, checkerboard_size, square_size, camera_matrix, dist_coeffs, refine = True ):
    """
    Finds the 3D pose of a known checkerboard in the input image.
    
    :param image: Input image (BGR or grayscale)
    :param checkerboard_size: Tuple (rows, cols) specifying the number of inner corners in the checkerboard
    :param square_size: The size of a single square in world units (e.g., mm or cm)
    :param camera_matrix: Camera intrinsic matrix (3x3)
    :param dist_coeffs: Distortion coefficients (1x5 or 1x8 array)
    :return: (success, rvec, tvec), where success is a boolean, rvec is the rotation vector, and tvec is the translation vector

    Source: https://chatgpt.com/canvas/shared/67ad397190148191bc0b41c33c32bd28
    """

    # Prepare object points (3D points in real-world coordinates)
    objp = np.zeros( (checkerboard_size[0] * checkerboard_size[1], 3), np.float32 )
    objp[:,:2] = np.mgrid[0:checkerboard_size[1], 0:checkerboard_size[0]].T.reshape( -1, 2 )
    objp *= square_size  # Scale by square size
    
    # Convert to grayscale if necessary
    if len( image.shape ) == 3:
        gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    else:
        gray = image
    
    # Find the checkerboard corners
    success, corners = cv2.findChessboardCorners( gray, checkerboard_size, None )
    
    if success:
        # Refine corner positions for better accuracy
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001, )
        )
        
        # Solve PnP to get initial rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP( objp, corners, camera_matrix, dist_coeffs )
        
        if refine:
            # Refine the pose using Levenberg-Marquardt optimization
            rvec, tvec = cv2.solvePnPRefineLM( objp, corners, camera_matrix, dist_coeffs, rvec, tvec )

        print( rvec.shape, tvec.shape )
        
        return success, rvec, tvec
    
    return False, None, None


########## MAIN ####################################################################################


##### Modes ###############################################################
_GET_DATA = False
_VIZ_DATA = True
_TEST_PCD = False
_CHKR_CAM = False
_CHKR_XFM = False


##### Settings ############################################################
_DATA_FIL = "data/CameraTSData.pkl"
# _DATA_FIL = "data/CameraCheckerData.pkl"
_TRANSFRM = True
_CHKR_DIM = (8,8,)
_CHKR_SIZ = 0.020


##### Main ################################################################
if __name__ == "__main__":

    poseSeq = [ _SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4, ]

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
                        geoLst.extend( cpcd_geo( obj, div = 5 ) )
        vispy_geo_list_window( geoLst )



    ##### Alternate PCD Generation ########################################

    if _TEST_PCD:
        rs  = RealSense( device_serial = "126122270157" )
        rs.initConnection()
        # rs  = RealSense()
        pcd, _ = rs.getPCD()
        print( type( pcd.colors ) )
        print( type( pcd.points ) )
        print( np.asarray( pcd.colors ).shape )
        print( np.asarray( pcd.points ).shape )
        pcd = rs.getPCD_alt()



    ##### Collect Checkerboard Data #######################################

    if _CHKR_CAM:

        poseSeq = [ _SAFE_POSE, _POSE_1, _POSE_2, _POSE_3, _POSE_4, ]

        try:
            set_object_env()
            set_camera_env()

            ctrl = MinController()
            ctrl.start()

            data = ctrl.images_at_poses( poseSeq )

            with open( _DATA_FIL, 'wb' ) as f:
                pickle.dump( data, f )

            ctrl.shutdown()

        except KeyboardInterrupt:
            ctrl.shutdown()



    ##### Calibrate Camera Transform ######################################

    if _CHKR_XFM:
        pass



    ##### CRASH OUT #######################################################
    os.system( 'kill %d' % os.getpid() ) 