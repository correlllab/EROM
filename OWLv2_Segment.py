""" Make it as simple as possible """
########## INIT ####################################################################################
### Standard ###
import sys, gc, time, traceback, warnings
now = time.time
from copy import deepcopy
from collections import defaultdict
from uuid import uuid4

# import torch
# torch.cuda.empty_cache()

# import torch
# import sam2
# from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

### Special ###
import numpy as np

### MAGPIE ###
from magpie_control.poses import vec_unit
from magpie_perception import pcd
from magpie_control import realsense_wrapper as real
from magpie_perception.label_owlv2 import LabelOWLv2

### ASPIRE ###
from aspire.env_config import env_var, env_sto
from aspire.symbols import CPCD


########## PERCEPTION SETTINGS #####################################################################

_VERBOSE = 1
_QUERIES = [ 
    # {'query': "a photo of a violet block", 'abbrv': "vio", },
    {'query': "a photo of a blue block"  , 'abbrv': "blu", },
    # {'query': "a photo of a red block"   , 'abbrv': "red", },
    {'query': "a photo of a yellow block", 'abbrv': "ylw", },
    {'query': "a photo of a green block" , 'abbrv': "grn", },
    # {'query': "a photo of a orange block", 'abbrv': "orn", },
]



########## ENVIRONMENT #############################################################################

def set_perc_env():
    """ Set perception params """
    
    env_sto( "_RSC_VIZ_SCL"  , 1000 ) 
    
    env_sto( "_OWL2_TOPK"    , 3     )
    env_sto( "_OWL2_THRESH"  , 0.005 )
    env_sto( "_OWL2_CPU"     , False )
    env_sto( "_OWL2_PATH"    , "google/owlv2-base-patch16-ensemble" ) 

    env_sto( "_SEG_MAX_HITS"    , 50     ) 
    env_sto( "_SEG_MAX_FRAC"    ,  0.05  ) 
    env_sto( "_SEG_SCORE_THRESH",  0.100 ) # 0.025 # 0.075 # 0.100
    env_sto( "_SEG_IOU_THRESH"  ,  0.750 )



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



########## SAM2 WRAPPER ############################################################################

class SAM2:
    """ Simplest SAM2 wrapper for bbox prompts """

    def __init__( self ):
        self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

    def predict(self, img, bbox):
        # Suppress warnings during the prediction step
        self.sam_predictor.set_image( img )
        sam_mask = None
        sam_scores = None
        sam_logits = None
        with warnings.catch_warnings():
            warnings.simplefilter( "ignore", category = UserWarning )
            sam_mask, sam_scores, sam_logits = self.sam_predictor.predict(box = bbox)
        sam_mask = np.all( sam_mask, axis = 0 )
        return sam_mask, sam_scores, sam_logits
    

    def __str__( self ):
        return f"SAM2: {self.sam_predictor.model.device}"
    
    def __repr__( self ):
        return self.__str__()



########## PERCEPTION WRAPPER ######################################################################


class Perception_OWLv2:
    """ Perception service based on OWLv2 """

    def __init__( self ):
        self.rsc : real.RealSense   = None
        self.label_vit : LabelOWLv2 = None 
        set_perc_env()


    def start_vision( self ):
        try:
            self.rsc = real.RealSense()
            self.rsc.initConnection()
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
        print( pnts.shape )
        
        print( f"{center=}" )

        # pose_vector = [center[0], center[1], center[2], 3.14, 0, 0]
        # HACK: HARDCODED ORIENTATION
        # FIXME: GET THE "ACTUAL" ORIENTATION VIA ICP
        pose_vector = np.eye(4)
        for i in range(3):
            pose_vector[i,3] = center[i]
        return pose_vector.reshape( (16,) ).tolist()
    

    def calculate_area( self, box ):
        """Calculates the area of the bounding box."""
        return abs(box[3] - box[1]) * abs(box[2] - box[0])


    def filter_by_area( self, tolerance, box, total_area ):
        """Filters the bounding box by area."""
        area = self.calculate_area(box)
        return abs(area / total_area) <= tolerance


    def bound( self, query, abbrevq ):
        """Bounds the given query with the OWLViT model."""
        _, rgbd_image = self.rsc.getPCD()
        image = np.array( rgbd_image.color )
        depth = np.array( rgbd_image.depth )

        # print( f"Image shape: {image.shape}", flush=True, file=sys.stderr )

        self.label_vit.set_threshold( env_var("_OWL2_THRESH") )

        _, _, scores, labels = self.label_vit.label( image, query, abbrevq, topk = True, plot = False )

        rtnHits = list()
        imgID   = str( uuid4() )
        for i in range( len( scores ) ):
            # if (scores[i] >= env_var("_SEG_SCORE_THRESH")) and \
            # self.filter_by_area( 
            #     env_var("_SEG_MAX_FRAC"), 
            #     self.label_vit.sorted_labeled_boxes_coords[i][0], 
            #     image.shape[0]*image.shape[1] 
            # ):
            if (scores[i] >= env_var("_SEG_SCORE_THRESH")):
                coords  = self.label_vit.sorted_boxes[i]
                indices = [int(c) for c in coords]
                rtnHits.append({
                    'bbox'   : coords,
                    'bboxi'  : indices,
                    'score'  : scores[i],
                    'label'  : labels[i],
                    'image'  : image[indices[1]:indices[3], indices[0]:indices[2]].copy(),
                    'query'  : query,
                    'abbrv'  : abbrevq,
                    'shotID' : imgID,
                })
            if len( rtnHits ) >= env_var("_SEG_MAX_HITS"):
                break

        return {
            'id'   : imgID,
            'rgbd' : rgbd_image,
            'image': image,
            'depth': depth,
            'hits' : rtnHits,
        }
    
    
    def segment( self, queries : list[dict] ) -> tuple[list[dict], list[dict]]: 
        """ Get poses from the camera """

        def mask_ray( mask : np.ndarray, bbox : np.ndarray ):
            """ Project a ray through the center of the mask """
            rows   = mask.shape[0]
            rwHf   = rows / 2
            cols   = mask.shape[1]
            clHf   = cols / 2
            cntr2d = np.zeros( 2 )
            count  = 0.0
            # Xlen   = np.sin( np.radians( env_var("_D405_FOV_H_DEG")/2.0 ) ) 
            # Ylen   = np.sin( np.radians( env_var("_D405_FOV_V_DEG")/2.0 ) ) 
            Xlen   = np.tan( np.radians( env_var("_D405_FOV_H_DEG")/2.0 ) ) 
            Ylen   = np.tan( np.radians( env_var("_D405_FOV_V_DEG")/2.0 ) ) 
            for j in range( bbox[1], min(bbox[3]-1, rows) ):
                for k in range( bbox[0], min(bbox[2]-1, cols) ):
                    # print( j,k )
                    frac_jk =  mask[j,k]
                    cntr2d  += np.array( [(k-clHf)/clHf,(j-rwHf)/rwHf] ) * frac_jk
                    count   += frac_jk
            cntr2d /= count
            return vec_unit( [cntr2d[0]*Xlen, cntr2d[1]*Ylen, 1.0] )
            

        rtnObjs  = list()
        metadata = {
            'input'  : dict(),
            'hits'   : list(),
        }

        try:

            ### Query the VLM ###

            for q in queries:

                query  = q['query']
                abbrv  = q['abbrv']
                result = self.bound( query, abbrv )

                metadata['input'][ result['id'] ] = {
                    'query': query, 'abbrv': abbrv, 
                    'image': result['image'].copy(), 
                    'depth': result['depth'].copy(),
                    'rgbd' : result['rgbd'], 
                    't'    : now(),
                }
                metadata['hits'].extend( deepcopy( result['hits'] ) )


            ### Get CPCDs from the Masks ###
            # rgbds = [result['rgbd'] for result in metadata]

            for hit_i in metadata['hits']:
                img_i = metadata['input'][ hit_i['shotID'] ]['image'].copy()

                sam_mask, _, _ = self.sam_predictor.predict( 
                    img_i, 
                    np.array( hit_i['bboxi'] ) 
                )
                samCount = (sam_mask > 0.1).sum()
                print( f"SAM2 Mask Dims: {sam_mask.shape}, Mask Count: {samCount}, Image Dims: {img_i.shape}" )
                
                if 100 < samCount < 12000:
                    mask_i = sam_mask.copy()
                else:
                    mask_i = bbox_to_mask( img_i.shape, hit_i['bboxi'] )

                ray_i = mask_ray( mask_i, hit_i['bboxi'] )

                if np.sum( mask_i ) < 100:
                    print( "MASK ERROR" )

                cpcd = None

                try:

                    print( type( metadata['input'][ hit_i['shotID'] ]['rgbd'] ) )

                    # _, cpcd = pcd.get_masked_cpcd( rgbds[0], hit_i['mask'], self.rsc, NB = 5 )
                    _, cpcd = pcd.get_masked_cpcd( 
                        metadata['input'][ hit_i['shotID'] ]['rgbd'], 
                        mask_i, 
                        self.rsc, 
                        NB = 50 
                    )

                except Exception as e:
                    print( f"Segmentation error: {e}", flush = True, file = sys.stderr )
                    raise e

                if len( np.asarray( cpcd.points ) ):

                    item = {
                        ## Updated ##
                        'Score'      : [hit_i['score'],],
                        'Probability': defaultdict( give_0 ),
                        'Count'      : 1,
                        ## Frozen ##
                        'bbox'       : hit_i['bbox'],
                        'Pose'       : self.get_pcd_pose( cpcd ),
                        'Time'       : now(),
                        'CPCD'       : { 'points' : np.asarray( cpcd.points ).copy(),
                                         'colors' : np.asarray( cpcd.colors ).copy(), },
                        'shotID'     : hit_i['shotID'],
                        'camRay'     : ray_i,
                    }
                    item['Probability'][ hit_i['abbrv'] ] = hit_i['score']
                    rtnObjs.append( item )

            # These don't pickle!
            for k in metadata['input'].keys():
                del metadata['input'][k]['rgbd']
                
            return rtnObjs, metadata

        except Exception as e:
            print(f"Error building model: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            raise e
        
        except KeyboardInterrupt as e:
            print( f"\n`segment` was stopped by user: {e}\n", flush=True, file=sys.stderr )
            raise e