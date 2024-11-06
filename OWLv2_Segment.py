""" Make it as simple as possible """
########## INIT ####################################################################################
### Standard ###
import sys, gc, time, traceback
now = time.time

### Special ###
import numpy as np
import open3d as o3d

### MAGPIE ###
from magpie_perception import pcd
from magpie_control import realsense_wrapper as real
from magpie_perception.label_owlv2 import LabelOWLv2

### ASPIRE ###
from aspire.env_config import env_var, env_sto


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
    env_sto( "_OWL2_TOPK"   , 3     )
    env_sto( "_OWL2_THRESH" , 0.005 )
    env_sto( "_OWL2_CPU"    , False )
    env_sto( "_OWL2_PATH"   , "google/owlv2-base-patch16-ensemble" ) 
    env_sto( "_RSC_VIZ_SCL" , 1000 ) 



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
    

    def bound( self, query, abbrevq ):
        """Bounds the given query with the OWLViT model."""
        _, rgbd_image = self.rsc.getPCD()
        image = np.array( rgbd_image.color )

        self.label_vit.set_threshold( env_var("_OWL2_THRESH") )

        _, boxes, scores, labels = self.label_vit.label( image, query, abbrevq, topk = True, plot = False )

        rtnHits = list()
        for i, box_i in enumerate( boxes ):
            rtnHits.append({
                'bbox'  : box_i,
                'score' : scores[i],
                'label' : labels[i],
            })

        return rgbd_image, image, rtnHits
    
    
    def segment( self, queries : list[dict] ) -> list[dict]: 
        """ Get poses from the camera """

        rtnObjs = list()

        try:

            for q in queries:

                query = q['query']
                abbrv = q['abbrv']

                rgbd, image, rtnHits = self.bound( query, abbrv )

                print( f"Obtained {len(rtnHits)} boxes!" )


                for i, hit_i in enumerate( rtnHits ):

                    cpcd = None

                    try:
                        _, cpcd, _, _ = pcd.get_segment(
                            hit_i['bbox'],
                            i,
                            rgbd,
                            self.rsc,
                            type      = "box",
                            method    = "iterative",
                            display   = False,
                            viz_scale = env_var("_RSC_VIZ_SCL")
                        )

                    except Exception as e:
                        print(f"Segmentation error: {e}", flush=True, file=sys.stderr)
                        raise e
                    
                    rtnObjs.append({
                        'Probability': hit_i['score'],
                        'Pose'       : self.get_pcd_pose( cpcd ),
                        'Count'      : 1,
                        'Time'       : now(),
                        'CPCD'       : { 'points' : np.asarray( cpcd.points ).copy(),
                                         'colors' : np.asarray( cpcd.colors ).copy(), }
                    })
                
            return rtnObjs

        except Exception as e:
            print(f"Error building model: {e}", flush=True, file=sys.stderr)
            traceback.print_exc()
            raise e
        
        except KeyboardInterrupt as e:
            print( f"\n`build_model` was stopped by user: {e}\n", flush=True, file=sys.stderr )
            raise e