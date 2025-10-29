import os, pickle, time, subprocess, math
now = time.time 
from collections import deque, defaultdict
from datetime import datetime
from copy import deepcopy
from random import choice

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as patches

import numpy as np
from skimage.measure import label # python3.10 -m pip install scikit-image --user
import cv2

from utils import deep_copy_memory_list, snap_z_to_nearest_block_unit_above_zero
from OWLv2_Segment import mask_ray_realsense
from homog_utils import posn_from_xform, diff_mag
from Geometry import closest_ray_points

from aspire.env_config import env_var, set_camera_env, set_object_env
from aspire.symbols import ( ObjPose, GraspObj, euclidean_distance_between_symbols, extract_pose_as_homog )


set_object_env()
set_camera_env()



########## LOGGER ##################################################################################

class LogPickler:
    """ Save Recordings as PKL files """

    def open_file( self ):
        """ Set the name of the current file """
        dateStr     = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.outNam = f"{self.prefix}_{dateStr}.pkl"
        if (self.outFil is not None) and (not self.outFil.closed):
            self.outFil.close()
        self.outFil = open( os.path.join( self.outDir, self.outNam ), 'wb' )


    def __init__( self, prefix = "Data-Log", outDir = None ):
        """ Set the file `prefix` and open a file """
        self.prefix = str( prefix ) # ------------------------- String to prepend to output filename
        self.outDir = outDir if (outDir is not None) else '.' # Root dir for saved data
        self.log    = deque() # ------------------------------- Actual Data
        self.outFil = None # ---------------------------------- Output file handle
        # WARNING: NOTHING IS DONE WITH THE FILE UNTIL THE END!
        self.open_file() # Actually create the file


    def dump_to_file( self, openNext = False ):
        """ Write all data lines to a file """
        if len( self.log ):
            if (self.outFil is None) or self.outFil.closed:
                self.open_file()
            pickle.dump( list( self.log ), self.outFil )
            self.outFil.close()
            self.log = deque()
        if openNext:
            self.open_file()

    
    def append( self, datum = None, msg = None ):
        """ Add an item to the log """
        self.log.append( {
            't'    : now(), # Timestamp
            'msg'  : msg, # - Datum Category, or Any string that makes this Searchable
            'data' : datum, # The Datum
        } )


    def visualize_last_segmentation( self, Nlast = None ):
        """ Overlay all the camera shots with the segmentations """
        _SEG_TAG = 'ObsMeta' # 'meta'
        _DAT_DIR = 'data'
        NdataPts = len( self.log )
        if Nlast is None:
            Nlast = min( env_var("_N_PERC_SHOTS"), env_var("_N_SEARCH_SHOTS") )
        # 0. Fetch the actual data
        datum = None
        count = 0
        for i in range( NdataPts ):
            datum_i = self.log[-i]
            if datum_i['msg'] == _SEG_TAG:
                count += 1
                if datum is None:
                    datum = datum_i
                else:
                    datum['data']['input'].update( datum_i['data']['input'] )
            if count >= Nlast:
                break

        if datum is not None:
            vizDct = dict()
            # 1. Copy Images
            for k, v in datum['data']['input'].items():
                vizDct[k] = {
                    'img': v['image'].copy(),
                    'seg': deque(),
                }
            # 2. Associate hits with images and brighten masks
            for hit in datum['data']['hits']:
                vizDct[ hit['shotID'] ]['seg'].append( hit )
                img    = vizDct[ hit['shotID'] ]['img']
                
                if 'mask' in hit:
                    msk = hit['mask'].copy()
                    for i in range( img.shape[0] ):
                        for j in range( img.shape[1] ):
                            if not (msk[i,j] > 0.001):
                                img[i,j,:] *= 0.25
            # 3. Draw Images w/ BB
            Nimg = len( vizDct )
            Ncol = 2
            Nrow = max( 1, int( math.ceil( Nimg/Ncol ) ) )
            rOne = (Nrow < 2)
            l    = 0
            fig, plots = plt.subplots( Nrow, Ncol, figsize = ( 4*Ncol, 3*Nrow, ) )
            print( Nimg, [Nrow,Ncol,], plots.shape )
            for k, v in vizDct.items():
                i = int(l / Ncol)
                j = int(l % Ncol)

                img : np.ndarray = v['img']
                if rOne:
                    ax  : Axes       = plots[l]
                else:
                    ax  : Axes       = plots[i,j]
                ax.imshow( img )

                for hit in v['seg']:
                    # Create a Rectangle patch
                    x      = hit['bbox'][0]
                    y      = hit['bbox'][1]
                    width  = hit['bbox'][2] - x
                    height = hit['bbox'][3] - y
                    rect   = patches.Rectangle( (x, y), width, height, linewidth=2, 
                                                edgecolor = hit['abbrv'][:1], 
                                                facecolor = 'none')
                    # Add the patch to the axes
                    ax.add_patch( rect )
                    

                    # Add score and label text
                    text = f"({hit['score']:.2f})"
                    ax.text(x, y, text, color='white', bbox=dict(facecolor='red', alpha=0.5))

                l += 1
            # Draw
            pdfPath = os.path.join( _DAT_DIR, f"Segmentations_{datum['t']}.pdf" )
            plt.savefig( pdfPath )
            # subprocess.call( ('xdg-open', pdfPath ) ) # WARNING: IS THIS BLOCKING?

        else:
            print( f"\nCould not find segmentation data in log of {NdataPts} entries!\n" )



########## POSE CHEATER ############################################################################

class PoseCheater:
    """ Fudge the `Memory` such that things are where they should be """

    def __init__( self, startSymbols = None, fix_labels = False, fix_poses = True ):
        """ Setup local memory """
        self.fixLabel = fix_labels
        self.fixPose  = fix_poses
        self.symbols  = deque( [startSymbols,] ) if isinstance( startSymbols, list ) else deque()
        self.beliefs  = deque() # WARNING: STATE LEAKAGE
        self.trouble  = False # WARNING: STATE LEAKAGE
        self.recover  = False # WARNING: STATE LEAKAGE


    def last_known_symbols( self ):
        """ Get last known symbol locations, even if we goofed last time """
        if self.trouble:
            return list()
        if len( self.symbols ):
            rtnSym = self.symbols[-1]
            index  = 2
            while ((not len( rtnSym )) and (index <= len( self.symbols ))):
                rtnSym = self.symbols[ -index ]
                index += 1
            return rtnSym 
            # return self.clean_frame( rtnSym )
        else:
            return list()
        

    def all_past_symbols( self ):
        rtnLst = list()
        for frame in self.symbols:
            rtnLst.extend( frame )
        return rtnLst
    

    def last_known_beliefs( self ):
        """ Get last known belief locations, even if we goofed last time """
        # WARNING: STATE LEAKAGE
        if self.trouble:
            return list()
        if len( self.beliefs ):
            rtnSym = self.beliefs[-1]
            index  = 2
            while ((not len( rtnSym )) and (index <= len( self.beliefs ))):
                rtnSym = self.beliefs[ -index ]
                index += 1
            return rtnSym
        else:
            return list()


    def log_symbols( self, symLst ):
        """ Store the most recent symbols """
        self.trouble = False
        self.symbols.append( deep_copy_memory_list( symLst ) )


    def log_beliefs( self, symLst ):
        """ Store the most recent symbols """
        self.beliefs.append( deep_copy_memory_list( symLst ) )


    def log_recovery( self, symLst = None ):
        """ It's all good now """
        self.trouble = False
        self.recover = True
        if symLst is not None:
            self.log_symbols( symLst )


    def log_successful_action( self, poseBgn, poseEnd ):
        """ Move the symbol to where the robot moved it """
        self.trouble = False
        print( f"Moved block by {euclidean_distance_between_symbols( poseBgn, poseEnd )}" )

        lastFrame = deep_copy_memory_list( self.last_known_symbols() )

        if len( lastFrame ):
            dMin = 6e10
            sCls : GraspObj = None
            for sym in lastFrame:
                d = euclidean_distance_between_symbols( sym, poseBgn )
                if (d < dMin) and (d < env_var("_BLOCK_SCALE")*0.75):
                    dMin = d
                    sCls = sym
            if sCls is not None:
                sCls.pose = ObjPose( poseEnd )
                self.symbols.append( lastFrame[:] )
            else:
                print( "`log_successful_action`: NO MATCH FOR THIS ACTION!" )

        else:
            print( "`log_successful_action`: NO FRAME TO MODIFY!" )
        

    def log_failed_action( self, poseBgn, poseEnd ):
        """ We done goofed, Erase symbol """
        self.trouble = True
        # self.symbols.append( list() )
        self.symbols = deque()
        # if (len( self.beliefs ) > 1):
        #     self.beliefs.pop()
        print( f"Could NOT move block by {euclidean_distance_between_symbols( poseBgn, poseEnd )}" )


    def clean_frame( self, objLst : list[GraspObj] ):
        """ Return a version of `objLst` with dupes removed """
        rtnLst = list()
        Nobj   = len( objLst )

        def p_collide_indices( qSym ):
            """ Did we already log a symbol at this location? """
            nonlocal rtnLst
            rtnDex = list()
            for i, rSym in enumerate( rtnLst ):
                d = euclidean_distance_between_symbols( qSym, rSym )
                if d < env_var('_BLOCK_SCALE')*0.65:
                    rtnDex.append(i)
            return rtnDex

        for i in range( Nobj-1 ):
            obj_i = objLst[i]
            cnflc = [obj_i,]
            for j in range( i+1, Nobj ):
                obj_j = objLst[j]
                if euclidean_distance_between_symbols( obj_i, obj_j ) < env_var('_BLOCK_SCALE')*0.65:
                    cnflc.append( obj_j )
            # WARNING: THIS ASSUMES THAT THE OLDEST SYMBOL WILL COLLIDE THE LEAST
            tCon = [obj.ts for obj in cnflc]
            tMin = min( tCon )
            mnDx = tCon.index( tMin )
            mnOb = cnflc[ mnDx ]
            cDcs = p_collide_indices( mnOb )
            if not len( cDcs ):
                rtnLst.append( mnOb )
            elif len( cDcs ) == 1:
                cObj = rtnLst[ cDcs[0] ]
                # WARNING: THIS ASSUMES THAT THE OLDEST SYMBOL WILL COLLIDE THE LEAST
                if mnOb.ts < cObj.ts:
                    rtnLst[ cDcs[0] ] = mnOb
        return rtnLst


    # def repair_symbol_poses( self, symLst : list[GraspObj], maxDiff = None ):
    def repair_symbol_poses( self, symLst : list[GraspObj], maxDiff = None ) -> list[GraspObj]:
        """ Adjust the positions of symbols to their last """
        lastFrame : list[GraspObj] = self.last_known_symbols()
        rtnSym = list()
        lSet   = set([])
        cSet   = set([])
        dlta   = False
        if maxDiff is None:
            maxDiff = 4.0 * env_var('_BLOCK_SCALE')

        def p_collide_return( qSym ):
            """ Did we already log a symbol at this location? """
            nonlocal rtnSym
            for rSym in rtnSym:
                # if euclidean_distance_between_symbols( qSym, rSym ) < env_var('_BLOCK_SCALE')*0.75:
                if euclidean_distance_between_symbols( qSym, rSym ) < env_var('_BLOCK_SCALE')*0.65:
                    return True
            return False
        
        def dct_Mars() -> float:
            """ Distance to Mars in meters """
            return {  
                "d"  : 6e10,
                "ref": None,
            }
        
        def p_assigned( dRefDct : dict, keyLst : list ):
            for k in keyLst:
                try:
                    if dRefDct[k]['ref'] is None:
                        return False
                except KeyError:
                    return False
            return True
        
        def readings_match( prvLst : list[GraspObj], nowLst : list[GraspObj], dThresh_m : float = None ) -> list[list[GraspObj]]:
            if dThresh_m is None:
                dThresh_m = env_var("_BLOCK_SCALE")*0.75
            lookup  = defaultdict( dct_Mars )
            for sym_i in prvLst:
                # idn_i = id( sym_i )
                for sym_j in nowLst:
                    idn_j = id( sym_j )
                    d_ij  = euclidean_distance_between_symbols( sym_i, sym_j )
                    if (d_ij <= dThresh_m) and d_ij < lookup[ idn_j ]['d']:
                        lookup[ idn_j ] = {  
                            "d"  : d_ij,
                            "ref": sym_i,
                        }
            rtnZip = deque()
            for sym_j in nowLst:
                idn_j = id( sym_j )
                rtnZip.append([ sym_j, lookup[ idn_j ]['ref'],])
            return list( rtnZip )

        if self.fixLabel and self.fixPose:
            for j, lSym in enumerate( lastFrame ):
                lSet.add( lSym.label )
                rtnSym.append( lSym )
            for i, rSym in enumerate( symLst ):
                if rSym.label not in lSet:
                    lSet.add( rSym.label )
                    rtnSym.append( rSym )
                    dlta = True

        elif self.fixLabel:
            symZip = readings_match( lastFrame, symLst, maxDiff )
            for pair in symZip:
                sym_i = pair[0]
                sym_j = pair[1]
                if sym_j is not None:
                    print( f"Match!: {sym_i} <-- {sym_j}" )
                    sym_i.label = sym_j.label
                if env_var("_Z_SNAP_CHEAT"):
                    # WARNING: HACK
                    nuPose = extract_pose_as_homog( sym_i )
                    nuPose[2,3] = snap_z_to_nearest_block_unit_above_zero( nuPose[2,3] )
                    sym_i.pose.pose = nuPose
                rtnSym.append( sym_i )


        elif self.fixPose:
            print( "CHEAT OBJECTS:" )
            for j, lSym in enumerate( lastFrame ):
                print( f"\t{lSym}" )
            symZip = readings_match( lastFrame, symLst, maxDiff )
            for pair in symZip:
                sym_i = pair[0]
                sym_j = pair[1]
                if sym_j is not None:
                    print( f"Match!: {sym_i} <-- {sym_j}: {euclidean_distance_between_symbols(sym_i, sym_j)}" )
                    sym_i.pose = sym_j.pose
                if env_var("_Z_SNAP_CHEAT"):
                    # WARNING: HACK
                    nuPose = extract_pose_as_homog( sym_i )
                    nuPose[2,3] = snap_z_to_nearest_block_unit_above_zero( nuPose[2,3] )
                    # sym_i.pose.pose = nuPose
                    sym_i.pose = ObjPose( nuPose )
                rtnSym.append( sym_i )
        else:
            print( "NO CHEAT APPLIED!" )     
            for i, rSym in enumerate( symLst ):
                if env_var("_Z_SNAP_CHEAT"):
                    # WARNING: HACK
                    nuPose      = extract_pose_as_homog( rSym )
                    nuPose[2,3] = snap_z_to_nearest_block_unit_above_zero( nuPose[2,3] )
                    rSym.pose.pose = nuPose
            rtnSym = symLst[:]

        self.symbols.append( rtnSym )
        return rtnSym
        


########## GROUND TRUTH EXTRACTOR ##################################################################

##### Block Masks ######################################################### 

def red_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Red Block """
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    lower     = np.array( [120, 120, 120,] )  # Example: lower bound for RED
    upper     = np.array( [255, 255, 255,] ) # Example: upper bound for RED
    # Create a mask for blue color
    return cv2.inRange( hsv_image, lower, upper)


def blu_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Blue Block """
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    lower     = np.array( [200/2, 150,  37,]) # Example: lower bound for BLUE
    upper     = np.array( [255/2, 255, 255,]) # Example: upper bound for BLUE
    # Create a mask for blue color
    return cv2.inRange( hsv_image, lower, upper)


def grn_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Green Block """
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    lower     = np.array( [ 80/2, 150,  15,] ) # Example: lower bound for GREEN
    upper     = np.array( [190/2, 255, 255,] ) # Example: upper bound for GREEN
    # Create a mask for blue color
    return cv2.inRange( hsv_image, lower, upper)


def blk_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Black Block """
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    lower     = np.array( [  0,   0,  0,] ) # Example: lower bound for BLACK, NOTE: THIS ONE IS GOING TO BE DIFFICULT!
    upper     = np.array( [180, 255, 62,] ) # Example: upper bound for BLACK
    # Create a mask for blue color
    return cv2.inRange( hsv_image, lower, upper)


def wht_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the White Block """
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    lower     = np.array( [  0,   0, 170,] ) # Example: lower bound for WHITE
    upper     = np.array( [172, 111, 255,] ) # Example: upper bound for WHITE
    # Create a mask for blue color
    return cv2.inRange( hsv_image, lower, upper)


def cluster_mask_arr( arrMsk : np.ndarray ):
    """ Get cluster masks """
    clusters = deque()
    labeled_mask = label( arrMsk, connectivity=2)
    for cluster_id in np.unique( labeled_mask ):
        if cluster_id != 0:  # Exclude background
            cluster_mask = ( labeled_mask == cluster_id)
            clusters.append( cluster_mask )
    clusters = list( clusters )
    clusters.sort( key = lambda x: np.count_nonzero(x), reverse = True )
    return clusters


def get_nonzero_mask_bbox( mask, flatXY = False ):
    """ Calculates the bounding box of non-zero elements in a 2D NumPy array (mask)."""
    # Get the row and column indices of non-zero elements
    rows, cols = np.where( mask )

    if rows.size == 0:  # No non-zero elements found
        return None

    # Calculate the minimum and maximum row and column indices
    y_min = int( np.min( rows ) )
    y_max = int( np.max( rows ) )
    x_min = int( np.min( cols ) )
    x_max = int( np.max( cols ) )

    if flatXY:
        # This is how `mask_ray_realsense` expects it
        return [ x_min, y_min, x_max, y_max,]
    else:
        # This is how Gemini ordered it
        return [ [y_min, x_min,], [y_max, x_max,],]


def vec3f_as_column( posn ):
    """ Get the position as a 1-scale column vector """
    rtnCol = np.ones( (4,1,) )
    rtnCol[:3,0] = posn
    return rtnCol


##### "Ground Truth" Tracker ############################################## 

class OCV_State_Tracker:
    """ Use OpenCV to infer something closer to the "Ground Truth", Prefer plain JSON """
    
    _CLUST_MIN =  500 # 1000
    _DIST_MIN  =    0.070
    _DIST_MAX  =    1.250
    _SCAL_MIN  =    0.500
    _SCAL_MAX  = _SCAL_MIN + 1.0 
    _CRIT_M    = env_var("_BLOCK_SCALE") * 2.25 # 1.50 # 1.750

    def __init__( self ):
        """ Set up tracking """
        self.names    = list() #- Names of the objects req'd to solve the problem
        self.scenes   = deque() # Sequence of reconstruction data
        self.states   = deque() # Sequence of States
        self.changes  = deque() # Sequence of Transitions
        self.current  = dict() #- All data relating to the current state
        self.maskFunc = { # ----- Function lookup to segment out the blocks in the experiments
            "redBlock": red_block_mask,
            "grnBlock": grn_block_mask,
            "bluBlock": blu_block_mask,
            "blkBlock": blk_block_mask,
            "whtBlock": wht_block_mask,
        }


    def find_block_mask( self, blockName : str, imgArr : np.ndarray, depArr : np.ndarray, imgID : str = None ):
        """ Search for the block, I guess! """
        blcMsk = self.maskFunc[ blockName ]( imgArr )
        clstrs = cluster_mask_arr( blcMsk )
        pixMax = -6e10
        clstMx = None
        # print( depArr[0,0] )
        for clstr in clstrs:
            # Test 1: Sufficient Points 
            Npix_i = np.count_nonzero( clstr )
            print( f"Block mask of {Npix_i} points!" )
            if Npix_i < self._CLUST_MIN:
                break # We sorted clusters descending
            # Test 2: Reasonable distance
            count_i  = np.count_nonzero( depArr[clstr] )
            if count_i == 0:
                continue
            depMsk_i = depArr[clstr].sum() / count_i
            print( f"Block mask is {depMsk_i} away!" )
            if (depMsk_i < self._DIST_MIN) or (depMsk_i > self._DIST_MAX):
                continue
            # Test 3: Expected size
            bbox_i = get_nonzero_mask_bbox( clstr )
            # span_i = [bbox_i[2]-bbox_i[0], bbox_i[3]-bbox_i[1],]
            span_i = [bbox_i[1][0]-bbox_i[0][0], bbox_i[1][1]-bbox_i[0][1],]
            print( f"Span is {span_i}" )
            angl_i = [ np.deg2rad( (span_i[0]/imgArr.shape[0])*env_var("_D405_FOV_V_DEG") ),
                       np.deg2rad( (span_i[1]/imgArr.shape[1])*env_var("_D405_FOV_H_DEG") ), ] 
            print( f"Arc is {angl_i}" )
            dims_i = [ 2.0 * np.tan( angl_i[0]/2.0 ) * depMsk_i, 
                       2.0 * np.tan( angl_i[1]/2.0 ) * depMsk_i, ] 
            scal_i = np.array( dims_i ) / env_var("_BLOCK_SCALE")
            print( f"Scale is {scal_i} * {env_var('_BLOCK_SCALE')}" )
            if (self._SCAL_MIN <= scal_i[0] <= self._SCAL_MAX) and (self._SCAL_MIN <= scal_i[1] <= self._SCAL_MAX):
                if Npix_i > pixMax:
                    pixMax = Npix_i
                    clstMx = clstr
        return clstMx      


    def new_scene( self ):
        """ Init Empty State Reconstruction """
        if len( self.current ):
            self.scenes.append( deepcopy( self.current ) )
        self.current = {
            "labels": list(),
            "image" : dict(),
            "depth" : dict(),
            "rays"  : deque()
        }


    @staticmethod
    def make_ray() -> dict:
        """ Create a container for a ray """
        return {
            "camPose": None,
            "imageID": None,
            "label"  : None,
            "rayOrg" : None,
            "rayDir" : None,
        }


    def process_observation_data( self, obsData : dict, goalLabels : list[str], camPose : np.ndarray = None ):
        """ Transform observation data into information about the current state """
        if camPose is None:
            camPose = np.eye(4)
        inpt = obsData['input']
        iKey = choice( list( inpt.keys() ) )
        imag = inpt[ iKey ]['image']
        dpth = inpt[ iKey ]['depth']
        self.current['image'][ iKey ] = deepcopy( imag )
        self.current['depth'][ iKey ] = deepcopy( dpth )
        Nadd = 0
        for label in goalLabels:
            res = self.find_block_mask( label, imag, dpth )
            if res is not None:
                self.current['labels'].append( label )
                print( f"MASK FOUND for {label}!" )
                ray_i  = OCV_State_Tracker.make_ray()
                rayVec = mask_ray_realsense( get_nonzero_mask_bbox( res, flatXY = True ), res )
                rayVec = np.dot( camPose, vec3f_as_column( rayVec ) ).reshape( (-1,) )[:3]
                ray_i["camPose"] = camPose.copy()
                ray_i["imageID"] = iKey
                ray_i["label"  ] = f"{label}"
                ray_i["rayOrg" ] = posn_from_xform( camPose )
                ray_i["rayDir" ] = rayVec
                self.current['rays'].append( ray_i )
                Nadd += 1
        print( f"\nAdded {Nadd} rays!\n\n" )


    def process_ray_obs( self ):
        """ Stage 2: Process ray observations """
        if not len( self.current ):
            return None
        _VERBOSE = 1
        rtnGobs = deque()
        centers = deque()
        print( f"There are {len(self.current['rays'])} RAY observations" )

        ### Local Helper Functions ###

        def center_index( q ):
            """ Find the index of the closest matching center """
            nonlocal centers
            rtnIdx = -5.5
            dMin   = 6e10
            for i, center in enumerate( centers ):
                ctr  = center['point']
                dSep = diff_mag( q, ctr )
                if dSep < dMin:
                    dMin = dSep
                    if dSep <= env_var("_BLOCK_SCALE")*0.80:
                        rtnIdx = i
            return rtnIdx

        ### For every pair of rays, Attempt to find an intersection ###

        rayItm = list( self.current['rays'] )
        Nrays  = len( self.current['rays'] )
        

        for i in range( Nrays-1 ):
            item_i = rayItm[i]
            for j in range( i+1, Nrays ):
                item_j = rayItm[j]

                if item_i['label'] != item_j['label']:
                    continue 

                pnt_ij, pnt_ji, center = closest_ray_points( 
                    item_i['rayOrg'], 
                    item_i['rayDir'], 
                    item_j['rayOrg'], 
                    item_j['rayDir'], 
                )
                if center is None:
                    print( "NO intersection!" )
                    continue
                elif (diff_mag( center, item_i['rayOrg'] ) <= env_var("_BLOCK_SCALE")*1.25) and (diff_mag( item_j['rayOrg'], center ) <= env_var("_BLOCK_SCALE")*1.25):
                    print( "Intersection at ORIGIN!" )
                    continue

                if _VERBOSE:
                    print( 
                        [i,j,],
                        item_i['rayOrg'], 
                        item_i['rayDir'], 
                        item_j['rayOrg'], 
                        item_j['rayDir']
                    )

                if diff_mag( pnt_ij, pnt_ji ) <= self._CRIT_M:
                    print( f"Log center {center} for separation {diff_mag( pnt_ij, pnt_ji )}" )
                    idx_ij  = center_index( center )
                    pair_ij = [item_i, item_j,]
                    if idx_ij > -1:
                        centers[ idx_ij ]['point'] = np.add( centers[ idx_ij ]['point'], center ) / 2.0
                        centers[ idx_ij ]['obs'].extend( pair_ij )
                    else:
                        centers.append( {
                            'point' : np.array( center ),
                            'obs'   : deque( pair_ij ),
                            'label' : item_i['label'],
                        } )
                else:
                    print( f"NO intersection for separation of {diff_mag( pnt_ij, pnt_ji )}/{self._CRIT_M}" )

        ### Construct an object for each intersection ###
        
        print( f"There {len(centers)} loci to evaluate!" )
        for ctrDct in centers:
            pnt_i    = ctrDct['point']
            objPose  = np.eye(4)
            objPose[0:3,3] = pnt_i
            obsDqu_i = ctrDct['obs']
            if len( obsDqu_i ):
                # Create reading
                rtnObj = GraspObj( 
                    label  = ctrDct['label'], 
                    pose   = ObjPose( objPose ), 
                    count  = len( obsDqu_i ), 
                )
                rtnGobs.append( rtnObj )

        return list( rtnGobs )


    def compare_states( self, ocvState : list[GraspObj], eromState : list[GraspObj] ):
        """ Return a list of object-wise differences between the two states """
        pass


    def episode_report( self, epPklPath : str ):
        """ Write a file that describes the results of each step of the episode, Prefer plain JSON """
        pass


    def diagnose_diff( self ):
        # FIXME: TRY TO DISCERN WHY THE DIFFERENCES OCCURRED????
        pass
