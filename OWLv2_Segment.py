""" Make it as simple as possible """
########## INIT ####################################################################################
### Standard ###
import sys, gc, time, traceback
now = time.time
from os import environ
from copy import deepcopy
from collections import defaultdict
from uuid import uuid4

# import torch
# torch.cuda.empty_cache()

import torch
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

### Special ###
import numpy as np
import open3d as o3d

### MAGPIE ###
from magpie_perception import pcd
from magpie_control import realsense_wrapper as real
from magpie_perception.label_owlv2 import LabelOWLv2

### ASPIRE ###
from aspire.env_config import env_var, env_sto
from aspire.utils import normalize_dist, diff_norm
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

    env_sto( "_SEG_MAX_HITS"    , 20     ) 
    env_sto( "_SEG_MAX_FRAC"    ,  0.05  ) 
    env_sto( "_SEG_SCORE_THRESH",  0.100 ) # 0.075
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

            if _VERBOSE:
                print( f"VLM STARTED", flush=True, file=sys.stderr )
        except Exception as e:
            if _VERBOSE:
                print( f"\nERROR initializing OWLv2: {e}\n", flush=True, file=sys.stderr )
            raise e
        
        try:
            print(f"{self.label_vit.model.device=}")
            self.sam_predictor = SAM2ImagePredictor.from_pretrained( "facebook/sam2-hiera-large" )
            print(f"{self.sam_predictor.model.device=}")
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
        center = point_cloud.get_center()

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
            if (scores[i] >= env_var("_SEG_SCORE_THRESH")) and \
            self.filter_by_area( 
                env_var("_SEG_MAX_FRAC"), 
                self.label_vit.sorted_labeled_boxes_coords[i][0], 
                image.shape[0]*image.shape[1] 
            ):
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

        hits     = list()
        metadata = list()
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
                match = False
                ## If Match, Then Update Object ##
                for obj in rtnObjs:
                    if bb_intersection_over_union( hit_i['bbox'], obj['bbox'] ) >= env_var("_SEG_IOU_THRESH"):
                        obj['Score'].append( hit_i['score'] )
                        obj['Probability'][ hit_i['abbrv'] ] += hit_i['score']
                        obj['Count'] += 1
                        match = True
                        break
                ## Else, New Obejct ##
                if not match:

                    cpcd = None

                    try:

                        self.sam_predictor.set_image( metadata['input'][ hit_i['shotID'] ]['image'] )
                        sam_box = np.array( hit_i['bbox'] )
                        sam_mask, _, _ = self.sam_predictor.predict( box = sam_box )
                        sam_mask = np.transpose( sam_mask, (1, 2, 0) )
                        hit_i['mask'] = np.asarray( sam_mask )

                        # _, cpcd = pcd.get_masked_cpcd( rgbds[0], hit_i['mask'], self.rsc, NB = 5 )
                        _, cpcd = pcd.get_masked_cpcd( 
                            metadata['input'][ hit_i['shotID'] ]['rgbd'], 
                            sam_mask, 
                            self.rsc, 
                            NB = 5 
                        )

                    except Exception as e:
                        print( f"Segmentation error: {e}", flush = True, file = sys.stderr )
                        raise e

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
                    }
                    item['Probability'][ hit_i['abbrv'] ] = hit_i['score']
                    rtnObjs.append( item )

            for obj in rtnObjs:
                obj['Probability'] = normalize_dist( obj['Probability'] )

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