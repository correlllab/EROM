""" Make it as simple as possible """
########## INIT ####################################################################################
### Standard ###
import sys, gc, time, traceback, warnings
now = time.time
from copy import deepcopy
from collections import defaultdict, deque
from uuid import uuid4

# import torch
# torch.cuda.empty_cache()

# import torch
# import sam2
# from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

### Special ###
import numpy as np
import cv2

### MAGPIE ###
from magpie_control.poses import vec_unit
from magpie_perception import pcd
from magpie_control import realsense_wrapper as real
from magpie_control.realsense_wrapper import get_oPCD_aabb_volume, MPCD
from magpie_perception.label_owlv2 import LabelOWLv2

### ASPIRE ###
from aspire.env_config import env_var, env_sto
from aspire.symbols import CPCD
from aspire.utils import normalize_dist


########## PERCEPTION SETTINGS #####################################################################

_VERBOSE = 1
_QUERIES = [ 
    # {'query': "a photo of a violet block", 'abbrv': "vio", },
    {'query': "a photo of a blue block"  , 'abbrv': "blu", },
    {'query': "a photo of a red block"   , 'abbrv': "red", },
    # {'query': "a photo of a yellow block", 'abbrv': "ylw", },
    {'query': "a photo of a green block" , 'abbrv': "grn", },
    # {'query': "a photo of a orange block", 'abbrv': "orn", },
]

_USE_ALT  = True
_NB_CLUST = 50



########## ENVIRONMENT #############################################################################

def set_perc_env():
    """ Set perception params """
    
    env_sto( "_RSC_VIZ_SCL"  , 1000 ) 
    
    env_sto( "_OWL2_TOPK"    , 3     )
    env_sto( "_OWL2_THRESH"  , 0.005 )
    env_sto( "_OWL2_CPU"     , False )
    env_sto( "_OWL2_PATH"    , "google/owlv2-base-patch16-ensemble" ) 

    env_sto( "_SEG_MAX_HITS"    , 50     ) 
    env_sto( "_SEG_SCORE_THRESH",  0.100 ) # 0.025 # 0.075 # 0.100



########## HELPER FUNCTIONS ########################################################################

def convert_to_CPCD( o3dCpcd ):
    """ Convert the points and colors to a CPCD """
    return CPCD(
        points = np.asarray( o3dCpcd.points ).copy(),
        colors = np.asarray( o3dCpcd.colors ).copy(),
    )


def bb_intersection( boxA, boxB ):
    """ Return true if the 2D bounding boxes intersect """
    # Author: Adrian Rosebrock, https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return [xA, yA, xB, yB,]
    

def bb_intersection_over_union( boxA, boxB ):
    """ Return IoU """
    # Author: Adrian Rosebrock, https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA, yA, xB, yB = bb_intersection( boxA, boxB )
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def give_0():
    """ Return Float Zero """
    return 0.0


def bbox_to_mask( maskShape, bbox ):
    """ Convert a bbox to a mask """
    mask = np.zeros( maskShape[:2] )
    mask[ bbox[1]:bbox[3], bbox[0]:bbox[2] ] = 1.0
    return mask


def mask_ray_realsense( bbox : np.ndarray, mask : np.ndarray = None  ):
    """ Project a ray through the center of the mask """
    cntr2d = np.zeros( 2 )
    count  = 0.0
    Xlen   = np.tan( np.radians( env_var("_D405_FOV_H_DEG")/2.0 ) ) 
    Ylen   = np.tan( np.radians( env_var("_D405_FOV_V_DEG")/2.0 ) ) 
    if mask is not None:
        rows = mask.shape[0]
        rwHf = rows / 2
        cols = mask.shape[1]
        clHf = cols / 2
        for j in range( bbox[1], min(bbox[3]-1, rows) ):
            for k in range( bbox[0], min(bbox[2]-1, cols) ):
                # print( j,k )
                frac_jk = mask[j,k]
                cntr2d += np.array( [(k-clHf)/clHf,(j-rwHf)/rwHf] ) * frac_jk
                count  += frac_jk
        cntr2d /= count
    else:
        cntr2d = np.array( [(bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0,] )
    return vec_unit( [cntr2d[0]*Xlen, cntr2d[1]*Ylen, 1.0] )


########## SAM2 WRAPPER ############################################################################

_BBOX_GAUSS_DIV = 4.0 # 3.5


# def sample_vec_normal_int( center, scale ) -> np.ndarray:
#     """ Sample independent normal coordinates """
#     Ndim      = len( center )
#     rtnCoords = [0 for _ in range(Ndim)]
#     for i in range( Ndim ):
#         rtnCoords[i] = int( np.random.normal( center[i], scale[i] ) )
#     return rtnCoords


# def sample_box_normal_int( bbox : list | np.ndarray, M : int = 1 ) -> np.ndarray:
#     """ Sample independent normal coordinates in a boudning box """
#     bbox   = np.array( bbox )
#     if len( bbox.shape ) < 2: # If we got the linear format, Convert to 2-row format
#         bbox = bbox.reshape( 2, -1 )
#     N      = bbox.shape[1]
#     center = [0 for _ in range(N)]
#     scale  = [0 for _ in range(N)]
#     for i in range(N):
#         center[i] = (bbox[1,i] + bbox[0,i]) / 2.0
#         scale[i]  = abs(bbox[1,i] - bbox[0,i]) / _BBOX_GAUSS_DIV
#     if M < 2:
#         return sample_vec_normal_int( center, scale )
#     else:
#         rtnArr = np.zeros( (M,N,) )
#         for i in range(M):
#             rtnArr[i] = sample_vec_normal_int( center, scale )
#         return rtnArr

def bbox_center( bbox : list | np.ndarray ) -> np.ndarray:
    """ Sample independent normal coordinates in a boudning box """
    bbox   = np.array( bbox )
    if len( bbox.shape ) < 2: # If we got the linear format, Convert to 2-row format
        bbox = bbox.reshape( 2, -1 )
    ctr = np.sum( bbox, axis = 0 )/2.0
    return [int(elem) for elem in ctr.tolist()]

class SAM2:
    """ Simplest SAM2 wrapper for bbox prompts """

    def __init__( self ):
        self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        self.NsamplePts    = 3

    def predict( self, img, bbox, useCache = False ):
        # Suppress warnings during the prediction step
        if not useCache:
            self.sam_predictor.set_image( img )
        sam_mask = None
        sam_scores = None
        sam_logits = None
        with warnings.catch_warnings():
            warnings.simplefilter( "ignore", category = UserWarning )
            sam_mask, sam_scores, sam_logits = self.sam_predictor.predict( 
                box = bbox,
                point_coords = [bbox_center( bbox ),],
                point_labels = np.ones( (1,) )
            )
        sam_mask = np.all( sam_mask, axis = 0 )
        return sam_mask, sam_scores, sam_logits
    

    def __str__( self ):
        return f"SAM2: {self.sam_predictor.model.device}"
    
    def __repr__( self ):
        return self.__str__()



########## PERCEPTION WRAPPER ######################################################################


class Perception_OWLv2:
    """ Perception service based on OWLv2 """
    _UNDISTORT = True
    _ADD_CONTR = True
    _DRW_EDGES = False

    def __init__( self ):
        set_perc_env()
        self.rsc : real.RealSense   = None
        self.label_vit : LabelOWLv2 = None 
        self.imgID : str        = None
        self.image : np.ndarray = None
        self.imgUD : np.ndarray = None
        self.depth : np.ndarray = None
        self.cloud : MPCD       = None
        self._SEG_SCORE_THRESH  = env_var("_SEG_SCORE_THRESH")
        

    def fetch_camera_model( self ):
        """ Get all the info we need to undistort """
        matx, coef, dims = self.rsc.getPinholeInstrinsics( distortion = True )
        self.matx = cv2.Mat( matx.intrinsic_matrix ) 
        self.coef = cv2.Mat( np.array( coef ) )
        self.dims = dims
        # https://claude.ai/public/artifacts/889c4ece-eaad-47ba-a193-f850f49a9a16
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix( self.matx, self.coef, self.dims, 1, self.dims )
        self.nMtx = new_camera_matrix


    def undistort( self, imgArr = None ):
        """ Use camera params to undistort the image """
        if imgArr is None:
            self.imgUD = np.asarray( cv2.undistort( self.image, self.matx, self.coef, None, self.nMtx ) )
            return self.imgUD
        else:
            return np.asarray( cv2.undistort( imgArr, self.matx, self.coef, None, self.nMtx ) )
            
    
    def contrastify( self, alpha : float = 1.5 ):
        """ Attempt to add contrast to the image """
        brighten = 0.0
        self.imgUD = np.asarray( cv2.convertScaleAbs( self.imgUD, alpha = alpha, beta = brighten ) )


    def get_Hough_edges( self ):
        """ Identify and mark straight edges in the image """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor( self.imgUD, cv2.COLOR_BGR2GRAY )
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur( gray, (5, 5), 0 )
        # Edge detection using Canny
        edges = cv2.Canny( blurred, 50, 150, apertureSize = 3 )
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,              # Distance resolution in pixels
            theta=np.pi/180,    # Angle resolution in radians
            threshold=100,      # Minimum number of votes
            minLineLength=50,   # Minimum line length
            maxLineGap=10       # Maximum gap between line segments
        )
        # List to store edge endpoints
        edge_endpoints = deque()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Draw the line on the result image
                cv2.line( self.imgUD, (x1, y1), (x2, y2), (0, 0, 0), 2)
                
                # Add endpoint pair to the list
                edge_endpoints.append( ((x1, y1,), (x2, y2,),) )
        return list( edge_endpoints )


    def scale_thresh_by_factor( self, factor ):
        """ Adjust the threshold by some factor """
        self._SEG_SCORE_THRESH *= factor
        # return self._SEG_SCORE_THRESH
        return self.label_vit.scale_thresh_by_factor( factor )


    def start_vision( self ):
        try:
            self.rsc = real.RealSense()
            self.rsc.initConnection()
            self.fetch_camera_model()
            if _VERBOSE:
                print( f"RealSense camera CONNECTED", flush=True, file=sys.stderr )
        except Exception as e:
            if _VERBOSE:
                print( f"\nERROR initializing RealSense: {e}\n", flush=True, file=sys.stderr )
            raise e
        
        try:
            self.label_vit = LabelOWLv2( 
                topk            = env_var("_OWL2_TOPK"), 
                score_threshold = env_var("_OWL2_THRESH"), 
                pth             = env_var("_OWL2_PATH"), 
                cpu_override    = env_var("_OWL2_CPU") 
            )
            print(f"{self.label_vit.model.device=}")
            self.label_vit.set_threshold( env_var("_OWL2_THRESH") )

            if _VERBOSE:
                print( f"VLM STARTED", flush=True, file=sys.stderr )
        except Exception as e:
            if _VERBOSE:
                print( f"\nERROR initializing OWLv2: {e}\n", flush=True, file=sys.stderr )
            raise e
        
        try:
            self.sam_predictor = SAM2()
            print(f"{self.sam_predictor.sam_predictor.model.device=}")
        except Exception as e:
            if _VERBOSE:
                print( f"\nERROR initializing SAM2: {e}\n", flush=True, file=sys.stderr )
            raise e
        
    
    def shutdown( self ):
        try:
            self.rsc.disconnect()
            if _VERBOSE:
                print( f"RealSense camera DISCONNECTED", flush=True, file=sys.stderr )
        except Exception as e:
            if _VERBOSE:
                print( f"\nERROR disconnecting RealSense: {e}\n", flush=True, file=sys.stderr )
            raise e
        
        try:
            del self.label_vit 
            self.label_vit = None
            gc.collect()
            if _VERBOSE:
                print( f"VLM SHUTDOWN", flush=True, file=sys.stderr )
        except Exception as e:
            if _VERBOSE:
                print( f"\nERROR cleaning OWLv2: {e}\n", flush=True, file=sys.stderr )
            raise e
        

    def get_pcd_pose( self, point_cloud ):
        """Gets the pose of the point cloud."""
        # center = point_cloud.get_center()
        pnts = np.asarray( point_cloud.points )
        if len( pnts ):
            center = np.mean( pnts, axis = 0 )
        else:
            center = np.zeros( 3 )
        # print( pnts.shape )
        
        # print( f"{center=}" )

        # pose_vector = [center[0], center[1], center[2], 3.14, 0, 0]
        # HACK: HARDCODED ORIENTATION
        # FIXME: GET THE "ACTUAL" ORIENTATION VIA ICP
        pose_vector = np.eye(4)
        for i in range(3):
            pose_vector[i,3] = center[i]
        return pose_vector.reshape( (16,) ).tolist()


    def bound( self, query, abbrevq, useCache = False ):
        """Bounds the given query with the OWLViT model."""

        if not useCache:
            self.imgID = str( uuid4() )
            self.cloud = self.rsc.getPCD_alt()
            rgbd_image = self.cloud.rgbd
            self.image = np.array( rgbd_image.color )
            self.depth = np.array( rgbd_image.depth )
            if self._UNDISTORT:
                self.undistort()
            if self._ADD_CONTR:
                self.contrastify( 1.25 ) # 1.5 # 1.75 # 2.0
            if self._DRW_EDGES:
                self.get_Hough_edges()

        else:
            rgbd_image = self.cloud.rgbd

        if (self._UNDISTORT or self._ADD_CONTR):
            _, _, scores, labels = self.label_vit.label( self.imgUD, query, abbrevq, topk = True, plot = False )
        else:
            _, _, scores, labels = self.label_vit.label( self.image, query, abbrevq, topk = True, plot = False )

        rtnHits = deque()
        
        for i in range( len( scores ) ):
            if (scores[i] >= self._SEG_SCORE_THRESH):
                coords  = self.label_vit.sorted_boxes[i]
                indices = [int(c) for c in coords]
                rtnHits.append({
                    'bbox'   : coords,
                    'bboxi'  : indices,
                    'score'  : scores[i],
                    'label'  : labels[i],
                    'image'  : self.image[indices[1]:indices[3], indices[0]:indices[2]].copy(),
                    'query'  : query,
                    'abbrv'  : abbrevq,
                    'shotID' : self.imgID,
                })
            if len( rtnHits ) >= env_var("_SEG_MAX_HITS"):
                break

        return {
            'id'    : self.imgID,
            'rgbd'  : rgbd_image,
            'image' : self.imgUD.copy() if (self._UNDISTORT or self._ADD_CONTR) else self.image.copy(),
            'depth' : self.depth.copy(),
            'mpcd'  : self.cloud,
            'hits'  : rtnHits,
        }
    

    def segment_cloud_w_SAM( self, img : np.ndarray, imgBBoxInt : list[list[int]], mpcd : MPCD,
                            #  loCount = 100, hiCount = 50000,
                             volEps = None, volThresh = None, useCache = False ):
        if (mpcd is None) or (not len( mpcd )):
            print( "`segment_cloud_w_SAM`: `mpcd` is None!" )
            return None
        if volEps is None:
            volEps    = 0.125 * env_var("_BLOCK_VOLUME")
        if volThresh is None:
            volThresh = 3.0 * env_var("_BLOCK_VOLUME")

        sam_mask, _, _ = self.sam_predictor.predict( 
            img, 
            np.array( imgBBoxInt ),
            useCache
        )
        samCount = (sam_mask > 0.1).sum()
        mask_i = sam_mask.copy()
        # if loCount < samCount < hiCount:
        #     mask_i = sam_mask.copy()
        # else:
        #     mask_i = bbox_to_mask( img.shape, imgBBoxInt )

        smCount = np.sum( mask_i )

        # if (hiCount < smCount) or (smCount < loCount):
        #     print( "MASK ERROR" )
        #     return None, None

        cpcd = mpcd.get_masked_cpcd( mask_i, NB = _NB_CLUST )
        if len( cpcd.points ):
            pcdVol = get_oPCD_aabb_volume( cpcd )

            if pcdVol > volThresh:
                print( f"CPCD TOO BIG: {pcdVol} > {volThresh}" )
                return None, None
            if pcdVol < volEps:
                print( f"CPCD TOO SMALL: {pcdVol} < {volEps}" )
                return None, None
            return cpcd, mask_i
        else:
            print( "CLOUD EMPTY" )
            return None, None
    
    
    def segment( self, queries : list[dict] ) -> tuple[list[dict], list[dict]]: 
        """ Get poses from the camera """

        rtnObjs  = deque()
        metadata = {
            'input'  : dict(),
            'hits'   : deque(),
        }

        try:

            ### Query the VLM ###

            for i, q in enumerate( queries ):

                query  = q['query']
                abbrv  = q['abbrv']
                # result = self.bound( query, abbrv, useCache = (i>0) )
                result = self.bound( query, abbrv, useCache = False )
                mpcd   = result['mpcd'] if ('mpcd' in result) else None

                metadata['input'][ result['id'] ] = {
                    'query': query, 'abbrv': abbrv, 
                    'image': result['image'].copy(), 
                    'depth': result['depth'].copy(),
                    'rgbd' : result['rgbd'], 
                    't'    : now(),
                    'mpcd' : mpcd,
                }
                # metadata['hits'].extend( deepcopy( result['hits'] ) )
                metadata['hits'].extend( result['hits'] )


            ### Get CPCDs from the Masks ###
            rtnDict = dict()
            lastID  = None

            for hit_i in metadata['hits']:
                bboxi_i = hit_i['bboxi']
                bbox_i  = hit_i['bbox']
                match   = False
                mtchKey = None
                intMax  = 0.0
                
                if len( rtnDict ):
                    print( f"BBox Intersection: ", end="", flush=True )
                    
                    for rK, rV in rtnDict.items():
                        bbox_j  = rV['bbox']
                        intrsct = bb_intersection_over_union( bbox_i, bbox_j )
                        print( f"{intrsct}, ", end="", flush=True )
                        # if (intrsct > 0.5) and (intrsct > intMax): # WARNING: ASSUMED PARAM!
                        if (intrsct > 0.75) and (intrsct > intMax): # WARNING: ASSUMED PARAM!
                            intMax  = intrsct
                            match   = True
                            mtchKey = rK
                    print()
                if match:
                    rtnDict[ mtchKey ]['Probability'][ hit_i['abbrv'] ] += hit_i['score']
                    print( f"Merge hit {mtchKey}, Overlap: {intMax}, Dist: {rtnDict[ mtchKey ]['Probability']}" )
                    if (rtnDict[ mtchKey ]['type'] == 'ray') and env_var("_RETRY_CLOUDS"):
                        img_i = metadata['input'][ hit_i['shotID'] ]['image'].copy()
                        print( "Match MISSING cloud!" )
                        cpcd, mask_i = self.segment_cloud_w_SAM( 
                            img_i, 
                            bboxi_i, 
                            metadata['input'][ hit_i['shotID'] ]['mpcd'],
                            useCache = False
                        )
                        if (cpcd is not None) and len( np.asarray( cpcd.points ) ):
                            print( "RECOVERED cloud!" )
                            pose_i    = self.get_pcd_pose( cpcd )
                            print( f"About to store PCD of {len( np.asarray( cpcd.points ) )} points from bbox {bboxi_i} @ {np.array(pose_i).reshape((4,4,))[0:3,3].reshape((3,))}!" )
                            boxRay_i  = mask_ray_realsense( hit_i['bboxi'], mask_i )
                            cloudPair = { 'points' : np.asarray( cpcd.points ).copy(),
                                        'colors' : np.asarray( cpcd.colors ).copy(), }
                            type_i    = 'cloud'
                            rtnDict[ mtchKey ]['type'] = type_i
                            rtnDict[ mtchKey ]['bbox'] = hit_i['bbox']
                            rtnDict[ mtchKey ]['Pose'] = pose_i
                            rtnDict[ mtchKey ]['CPCD'] = cloudPair
                        else:
                            print( "NO cloud!" )


                else:
                    print( f"No merge for max overlap {intMax} of bbox {bboxi_i}" )

                    img_i = metadata['input'][ hit_i['shotID'] ]['image'].copy()
                    # img_i = metadata['input'][ hit_i['shotID'] ]['image']

                    cpcd, mask_i = self.segment_cloud_w_SAM( 
                        img_i, 
                        bboxi_i, 
                        metadata['input'][ hit_i['shotID'] ]['mpcd'],
                        # useCache = repeat
                        useCache = False
                    )
                    if (cpcd is not None) and len( np.asarray( cpcd.points ) ):
                        pose_i    = self.get_pcd_pose( cpcd )
                        print( f"About to store PCD of {len( np.asarray( cpcd.points ) )} points from bbox {bboxi_i} @ {np.array(pose_i).reshape((4,4,))[0:3,3].reshape((3,))}!" )
                        boxRay_i  = mask_ray_realsense( hit_i['bboxi'], mask_i )
                        cloudPair = { 'points' : np.asarray( cpcd.points ).copy(),
                                      'colors' : np.asarray( cpcd.colors ).copy(), }
                        type_i    = 'cloud'
                        
                        
                    else:
                        boxRay_i  = mask_ray_realsense( hit_i['bboxi'] )
                        cloudPair = { 'points' : None,
                                      'colors' : None, }
                        type_i    = 'ray'
                        pose_i    = None

                    item = {
                        ## Updated ##
                        'Score'      : [hit_i['score'],],
                        'Probability': defaultdict( give_0 ),
                        'Count'      : 1,
                        ## Frozen ##
                        'type'       : type_i,
                        'bbox'       : hit_i['bbox'],
                        'Pose'       : pose_i,
                        'Time'       : now(),
                        'CPCD'       : cloudPair,
                        'shotID'     : hit_i['shotID'],
                        'camRay'     : np.array([0,0,1,]),
                        'boxRay'     : boxRay_i,
                    }
                    item['Probability'][ hit_i['abbrv'] ] = hit_i['score']
                    rtnDict[ uuid4() ] = item 

            for rK in rtnDict.keys():
                rtnDict[ rK ]['Probability'] = normalize_dist( rtnDict[ rK ]['Probability'] )
            rtnObjs = list( rtnDict.values() )

            # These don't pickle!
            for k in metadata['input'].keys():
                del metadata['input'][k]['rgbd']
                del metadata['input'][k]['mpcd']
                
            return list( rtnObjs ), metadata

        except Exception as e:
            print(f"Error building model: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            raise e
        
        except KeyboardInterrupt as e:
            print( f"\n`segment` was stopped by user: {e}\n", flush=True, file=sys.stderr )
            raise e