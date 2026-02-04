import os, pickle, time, traceback, math
now = time.time 
from collections import deque, defaultdict
from datetime import datetime
from copy import deepcopy
from random import choice

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as patches

import numpy as np
from scipy import ndimage
from skimage.measure import label # python3.10 -m pip install scikit-image --user
import cv2
import pyrealsense2 as rs
import open3d as o3d

from utils import deep_copy_memory_list, snap_z_to_nearest_block_unit_above_zero, JupyterPlotServer

from OWLv2_Segment import mask_ray_realsense
from homog_utils import posn_from_xform, diff_mag
from Geometry import closest_ray_points

from aspire.env_config import env_var, set_camera_env, set_object_env
from aspire.symbols import ( ObjPose, GraspObj, euclidean_distance_between_symbols, extract_pose_as_homog, extract_position,
                             p_symbol_inside_workspace_bounds )
from aspire.utils import diff_norm

from magpie_control.realsense_wrapper import MPCD



########## SETTINGS ################################################################################
set_object_env()
set_camera_env()

_RAM_SAVER = True


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
from scipy.ndimage import convolve

##### Helper Functions #################################################### 

def crash_out():
    """ End the program with Brutal Finality """
    print( "\n\n" )
    os.system( 'kill %d' % os.getpid() ) 


def extract_pose_from_str( poseStr : str ):
    """ Get the homogeneous coordinates from the string and ignore everything else """
    nstLst = list()
    depth  = 0
    numStr = ""
    row    = list()

    def store_num():
        """ Add the number to the row """
        nonlocal row, numStr, poseStr
        if len( numStr ):
            try:
                row.append( float( numStr.strip() ) )
            except ValueError as e:
                print( f"BAD: {e}" )
                print( numStr  )
                print( poseStr )
                crash_out()
        numStr = ""

    def store_row():
        """ Add the row to the array """
        nonlocal nstLst, row
        if len( row ):
            nstLst.append( row )
        row = list()

    for char in poseStr:
        if char == '[':
            depth += 1
        elif char == ']':
            if depth == 2:
                store_num()
            depth -= 1
            if depth == 1:
                store_row()
        elif char == ' ':
            if depth == 2:
                store_num()
        elif char == '\n':
            pass
        elif depth == 2:
            numStr += char
        else:
            pass
            # print( f"`extract_pose_from_str()`, BAD STATE:\n{char}\n{poseStr}\n" )

    try:
        return np.array( nstLst )
    except Exception as e:
        traceback.print_exc()
        print( f"BAD: {e}" )
        crash_out()

##### Block Masks ######################################################### 

def red_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Red Block """
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    lower     = np.array( [145, 120, 120,] )  # Example: lower bound for RED
    upper     = np.array( [179, 255, 255,] ) # Example: upper bound for RED
    # Create a mask for blue color
    return cv2.inRange( hsv_image, lower, upper)


def blu_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Blue Block """
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    nudge     = 0
    lower     = np.array( [ 95, 128, int(0.35*255),]) # Example: lower bound for BLUE
    upper     = np.array( [135, 255, int(1.00*255),]) # Example: upper bound for BLUE
    # Create a mask for blue color
    return cv2.inRange( hsv_image, lower, upper)


def grn_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Green Block """
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    lower     = np.array( [ 40, 150,  15,] ) # Example: lower bound for GREEN
    upper     = np.array( [ 95, 255, 255,] ) # Example: upper bound for GREEN
    # Create a mask for blue color
    return cv2.inRange( hsv_image, lower, upper)

# _BLK_HI = 75 # 80 # 85

def blk_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Black Block """
    _VERBOSE = 0 and (not _RAM_SAVER)
    jps      = None 
    if _VERBOSE:
        jps     = JupyterPlotServer()
    hsv_image = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    lower     = np.array( [int( 90/360*179), int(20/100*255), int(  9/100*255),] ) # Example: lower bound for BLACK, NOTE: THIS ONE IS GOING TO BE DIFFICULT!
    upper     = np.array( [int(210/360*179), int(77/100*255), int( 45/100*255),] ) # Example: upper bound for BLACK
    # Create a mask for blue color
    rtnMsk = cv2.inRange( hsv_image, lower, upper)
    if _VERBOSE:
        print( "BLACK MASK" )
        jps.arr_show( rtnMsk )
    return rtnMsk

# _WHITE_HUE_LO = int(180/360*179)
# _WHITE_HUE_HI = int(210/360*179)
# _WHITE_HUE_MD = int((_WHITE_HUE_LO+_WHITE_HUE_HI)/2)

# _WHITE_SAT_LO = int( 0/100*255)
# _WHITE_SAT_HI = int(50/100*255)
# _WHITE_SAT_MD = int((_WHITE_SAT_LO+_WHITE_SAT_HI)/2)

# _WHITE_VAL_LO = int( 50/100*255)
# _WHITE_VAL_HI = int(100/100*255)
# _WHITE_VAL_MD = int((_WHITE_VAL_LO+_WHITE_VAL_HI)/2)


_GRY_LO = np.array( [int(182/360*179), int(20/100*255), int( 80/100*255),] )
_GRY_HI = np.array( [int(210/360*179), int(40/100*255), int(100/100*255),] )
_WHT_LO = np.array( [int(  0/360*179), int( 0/100*255), int( 98/100*255),] )
_WHT_HI = np.array( [int(182/360*179), int(20/100*255), int(100/100*255),] )
_B_LVL  = 0.125 # 0.0625


def sobel_mask_intolerant( img : np.ndarray ) -> np.ndarray:
    """ Border mask with a hair trigger """
    gImg = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    # Apply Sobel operator
    sobelx = cv2.Sobel(gImg, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)  # Horizontal edges
    sobely = cv2.Sobel(gImg, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)  # Vertical edges
    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    # Convert to uint8
    gradient_magnitude = cv2.convertScaleAbs( gradient_magnitude )
    return (gradient_magnitude >= ( _B_LVL * 255)).astype("bool")


def gry_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Black Block """
    _VERBOSE = 0 and (not _RAM_SAVER)
    jps      = None 
    if _VERBOSE:
        jps     = JupyterPlotServer()
    hsv_image = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    lower     = _GRY_LO # Example: lower bound for BLACK, NOTE: THIS ONE IS GOING TO BE DIFFICULT!
    upper     = _GRY_HI # Example: upper bound for BLACK
    boxMsk    = cv2.inRange( hsv_image, lower, upper)
    brdrMsk   = sobel_mask_intolerant( img )
    if _VERBOSE:
        print( "GREY MASK" )
        jps.arr_show( boxMsk )
        print( "BORDER MASK" )
        jps.arr_show( brdrMsk )
    boxMsk = np.logical_and( boxMsk, ~brdrMsk )

    return boxMsk 


def wht_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the White Block """
    # Convert BGR to HSV
    _VERBOSE = 0 and (not _RAM_SAVER)
    jps = None
    if _VERBOSE:
        jps = JupyterPlotServer()
    hsv_image = cv2.cvtColor( img, cv2.COLOR_RGB2HSV )
    lower     = _WHT_LO # Example: lower bound for WHITE
    upper     = _WHT_HI # Example: upper bound for WHITE
    # Create a mask for blue color
    rtnMsk  = cv2.inRange( hsv_image, lower, upper)
    brdrMsk = sobel_mask_intolerant( img )
    rtnMsk  = np.logical_and( rtnMsk, ~brdrMsk )
    if _VERBOSE:
        print( "WHITE MASK" )
        jps.arr_show( rtnMsk )
    return rtnMsk


##### Mask Operations ##################################################### 

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
        return [ x_min, y_min, x_max, y_max,] # [xLo, yLo, xHi, yHi]
    else:
        # This is how Gemini ordered it
        return [ [y_min, x_min,], [y_max, x_max,],] # [[rowLo, colLo], [rowHi, colHi]]
    

def mask_density( mask : np.ndarray ):
    """ How much of the bbox is actually occupied by the mask? """
    bbox = get_nonzero_mask_bbox( mask )
    A_bb = abs(bbox[1][0] - bbox[0][0]) * abs(bbox[1][1] - bbox[0][1])
    c_bb = np.count_nonzero( mask )
    return c_bb / A_bb


def get_clumped_mask( binary_image : np.ndarray, N : int = 4 ):
    """ Returns a mask of pixels in a binary image that at least N neighbors (N-connectivity) """
    # Ensure the image is boolean for consistent behavior
    binary_image = binary_image.astype( bool )
    kernel_conn  = np.array([[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]])
    # Convolve the binary image with the kernel to count neighbors
    neighbor_counts = convolve( binary_image.astype(int), kernel_conn, mode = 'constant', cval = 0 )
    # Create the mask: pixels that are foreground AND have exactly 4 neighbors
    mask = (binary_image) & (neighbor_counts >= N)
    return mask


def clumped_density( mask : np.ndarray, N : int = 4 ):
    """ How much of the bbox is actually occupied by the mask? """
    bbox = get_nonzero_mask_bbox( mask )
    A_bb = abs(bbox[1][0] - bbox[0][0]) * abs(bbox[1][1] - bbox[0][1])
    c_bb = np.count_nonzero( get_clumped_mask( mask, N ) )
    if A_bb > 0.0:
        return c_bb / A_bb
    return 0.0


def grow_clumped_mask( binary_image : np.ndarray, N : int = 4, depth : int = 2 ):
    """ Expand all pixels with at least N neighbors (N-connectivity) by `depth` """
    clumpMask = get_clumped_mask( binary_image, N )
    # 2. Define the kernel (structuring element).
    # A kernel of all ones will "expand" the True regions to cover
    # all areas where the kernel overlaps with a True value in the original mask.
    kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=bool) # Kernel should typically be boolean or integer 1s and 0s for dilation
    # 3. Perform the binary dilation, `depth` times
    expanded_mask = clumpMask.copy()
    for _ in range( depth ):
        expanded_mask = ndimage.binary_dilation( expanded_mask, structure = kernel )
    return expanded_mask


def mask_clump_ratio( mask : np.ndarray, N : int = 4 ):
    """ What fraction of the mask pixels have at least N neighbors? """
    cM = get_clumped_mask( mask, N )
    Nc = np.count_nonzero( cM )
    Nt = np.count_nonzero( mask )
    return Nc / Nt


def mask_midline_ratio( mask : np.ndarray ):
    bbox   = get_nonzero_mask_bbox( mask )
    rowMid = int( (bbox[1][0] + bbox[0][0])/2 )
    colMid = int( (bbox[1][1] + bbox[0][1])/2 )
    Ntot   = (bbox[1][0] - bbox[0][0])+(bbox[1][1] - bbox[0][1])
    Nyes   = 0
    for j in range( bbox[0][1], bbox[1][1]+1 ):
        if mask[ rowMid ][j]:
            Nyes += 1    
    for i in range( bbox[0][0], bbox[1][0]+1 ):
        if mask[i][ colMid ]:
            Nyes += 1
    if Ntot > 0:
        return Nyes / Ntot
    else:
        return 0.0


def vec3f_as_column( posn ):
    """ Get the position as a 1-scale column vector """
    rtnCol = np.ones( (4,1,) )
    rtnCol[:3,0] = posn
    return rtnCol


"""
 Intrinsic of "Color" / 1280x720 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8}
  Width:      	1280
  Height:     	720
  PPX:        	635.127990722656
  PPY:        	356.310791015625
  Fx:         	641.381103515625
  Fy:         	640.520446777344
  Distortion: 	Inverse Brown Conrady
  Coeffs:     	-0.0538796000182629  	0.0600070655345917  	-1.4652669960924e-05  	0.000627028464805335  	-0.0188705865293741  
  FOV (deg):  	89.88 x 58.67

 Intrinsic of "Depth" / 1280x720 / {Z16}
  Width:      	1280
  Height:     	720
  PPX:        	633.974792480469
  PPY:        	356.181549072266
  Fx:         	634.66357421875
  Fy:         	634.66357421875
  Distortion: 	Brown Conrady
  Coeffs:     	0  	0  	0  	0  	0  
  FOV (deg):  	90.48 x 59.13
"""

_COLOR_MATX_1280x720 = {
    "fx":         641.381103515625,
    "fy":         640.520446777344,
    "cx":         635.127990722656,
    "cy":         356.310791015625,
    "distortion": [-0.0538796000182629, 0.0600070655345917, -1.4652669960924e-05, 0.000627028464805335, -0.0188705865293741,],
}

_DEPTH_MATX_1280x720 = {
    "fx":         634.66357421875,
    "fy":         634.66357421875,
    "cx":         633.974792480469,
    "cy":         356.181549072266,
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0,],
}


def rgbd_to_color( rawRGBDImage : np.ndarray ):
    """ Return the color portion of RGB-D """
    return rawRGBDImage[:,:,:3].copy()


def rgbd_to_depth( rawRGBDImage : np.ndarray ):
    """ Return the color portion of RGB-D """
    return rawRGBDImage[:,:,:-1].copy()


def color_depth_to_pointcloud( color_image : np.ndarray, depth_image : np.ndarray, intrinsics : dict | np.ndarray, 
                               distortion_color = None, distortion_depth = None, mask : np.ndarray = None ):
    """ Convert RGB-D images to a point cloud, https://claude.ai/public/artifacts/53bc4d27-e2b8-4fd7-b36f-c55e0d442a85 """

    print( color_image.shape )
    print( depth_image.shape )

    # Mask if requested
    if mask is not None:
        # 3. Expand the dimensions of the 2D mask to be compatible with the 3D array
        # We add a new axis at the end to match the third dimension of array_3d
        msk3 = mask[:, :, np.newaxis]
        color_image = np.where( msk3, color_image, 0 ) # color_image[ mask ]
        depth_image = np.where( mask, depth_image, 0 ) # depth_image[ mask ]

    # Parse intrinsics
    if isinstance( intrinsics, dict ):
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']
        camera_matrix = np.array( [[fx , 0.0, cx ,],
                                   [0.0, fy , cy ,],
                                   [0.0, 0.0, 1.0,]], dtype=np.float32)
    else:
        camera_matrix = intrinsics
        fx = camera_matrix[0,0]
        fy = camera_matrix[1,1]
        cx = camera_matrix[0,2]
        cy = camera_matrix[1,2]
    
    print( color_image.shape )
    print( depth_image.shape )

    h = depth_image.shape[0]
    w = depth_image.shape[1]
    
    # Undistort images if distortion coefficients are provided
    if distortion_color is not None:
        color_image = cv2.undistort( color_image, camera_matrix, distortion_color )
    if distortion_depth is not None:
        depth_image = cv2.undistort( depth_image, camera_matrix, distortion_depth )
    
    # ASSUMPTION: THIS IS JUST STACKED???
    rawRGBDImage = np.zeros( (h, w, 4) )
    rawRGBDImage[:,:,:3] = color_image 
    rawRGBDImage[:,:,-1] = depth_image

    # Create mesh grid of pixel coordinates
    u, v = np.meshgrid( np.arange(w), np.arange(h) )
    
    # Flatten arrays
    u     = u.flatten()
    v     = v.flatten()
    depth = depth_image.flatten()
    
    # Filter out invalid depth values (zero or negative)
    valid = depth > 0
    u     = u[valid]
    v     = v[valid]
    depth = depth[valid]
    
    # Convert pixel coordinates to 3D points
    # Using pinhole camera model: X = (u - cx) * Z / fx
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    # Stack into point cloud
    points = np.stack([x, y, z], axis=-1)
    print( f"Points Array with shape: {points.shape}" )
    
    # Extract corresponding colors (convert BGR to RGB)
    colors = color_image[ v.astype(int), u.astype(int) ]
    colors = cv2.cvtColor( colors.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB ).reshape(-1, 3)
    print( f"Points Array with shape: {points.shape}" )
    
    # return points, colors
    return MPCD( rawRGBDImage, points, colors )


def masked_rgbd_to_pointcloud( rawRGBDImage : np.ndarray, intrinsics : dict | np.ndarray, 
                               distortion_color = None, distortion_depth = None, mask : np.ndarray = None ):
    """ Mask the stacked RGB-D Image and calc a point cloud for it """
    color_image = rgbd_to_color( rawRGBDImage )
    depth_image = rgbd_to_depth( rawRGBDImage )
    return color_depth_to_pointcloud( color_image, depth_image, intrinsics, distortion_color, distortion_depth, mask )
    

def transform_mpcd( point_cloud : MPCD, xform : np.ndarray ):
    points = np.hstack( (point_cloud.xyzArr, np.ones( (point_cloud.xyzArr.shape[0],1,) )) )
    xPnts  = np.dot( xform, points.T )
    point_cloud.xyzArr = xPnts.T[:,:3]
    print( f"Transformed {len(point_cloud.xyzArr)} points!" ) 


def get_mpcd_pose( point_cloud : MPCD ):
    """ Gets the pose of the point cloud. """
    if len( point_cloud.xyzArr ):
        center = np.mean( point_cloud.xyzArr, axis = 0 )
    else:
        center = np.zeros( 3 )
    # HACK: HARDCODED ORIENTATION
    # FIXME: GET THE "ACTUAL" ORIENTATION VIA ICP
    pose = np.eye(4)
    pose[:3,3] = center
    return pose


##### "Ground Truth" Tracker ############################################## 

_CLUMP_POP_PX  = 7 # Number of neighbors to be considered part of a clump
_EXPAND_DEPTH  = 3
_MIN_PXL_DNSTY = 0.25
_MIN_MID_RATIO = 0.40
_N_MIN_POINTS  = 4000

class OCV_State_Tracker:
    """ Use OpenCV to infer something closer to the "Ground Truth", Prefer plain JSON """
    
    _CLUST_MIN = _N_MIN_POINTS # 125 # 250 # 500 # 750 # 1000
    _PCD_MIN   = _N_MIN_POINTS # 600 # 100
    _DIST_MIN  =    0.070
    _DIST_MAX  =    1.250
    _SCAL_MIN  =    0.350 # 0.350 # 0.500
    _SCAL_MAX  = (1.0 - _SCAL_MIN) + 1.0 
    _CRIT_M    = env_var("_BLOCK_SCALE") * 1.50 # 1.50 # 1.750 # 2.25
    _N_RETAIN  = 5

    def __init__( self, readFile : str = None ):
        """ Set up tracking """
        self.seq      = 0
        self.rFile    = readFile
        self.jps      = JupyterPlotServer()
        self.names    = list() #- Names of the objects req'd to solve the problem
        self.scenes   = deque() # Sequence of reconstruction data
        self.actions  = deque() # Sequence of Actions
        self.current  = dict() #- All data relating to the current state
        self.maskFunc = { # ----- Function lookup to segment out the blocks in the experiments
            "redBlock": { "func": red_block_mask, "grow": 0, "clump": 7 },
            "grnBlock": { "func": grn_block_mask, "grow": 0, "clump": 7 },
            "bluBlock": { "func": blu_block_mask, "grow": 0, "clump": 7 },
            "blkBlock": { "func": blk_block_mask, "grow": 2, "clump": 6 },
            "whtBlock": { "func": [gry_block_mask, wht_block_mask,], "grow": 4, "clump": 5 },
        }
        self.new_scene()


    def find_block_mask( self, blockName : str, imgArr : np.ndarray, depArr : np.ndarray ):
        """ Search for the block, I guess! """
        _VERBOSE = False and (not _RAM_SAVER)

        if isinstance( self.maskFunc[ blockName ]['func'], list ):
            blcMsk = np.zeros( imgArr.shape[:2] )
            for func in self.maskFunc[ blockName ]['func']:
                blcMsk = np.logical_or( blcMsk, func( imgArr ) ) 
        else:
            blcMsk = self.maskFunc[ blockName ]['func']( imgArr )
        
        if self.maskFunc[ blockName ]['grow'] > 0:
            blcMsk = grow_clumped_mask( blcMsk, self.maskFunc[ blockName ]['clump'], self.maskFunc[ blockName ]['grow'] )
        
        clstrs = cluster_mask_arr( blcMsk )
        pixMax = -6e10
        clstMx = None
        # print( depArr[0,0] )
        for clstr in clstrs:
            
            if _VERBOSE:
                print( f"Evaluate: {self.maskFunc[ blockName ]}" ) 
                self.jps.arr_show( clstr )
            
            # Test 1: Sufficient Points 
            Npix_i = np.count_nonzero( clstr )
            # Nclm_i = mask_clump_ratio( clstr, _CLUMP_POP_PX ) * Npix_i
            # Nclm_i = clumped_density( clstr, _CLUMP_POP_PX ) * Npix_i
            dnsty_i = clumped_density( clstr, _CLUMP_POP_PX+1 )
            ratio_i = mask_midline_ratio( clstr )
            if dnsty_i < _MIN_PXL_DNSTY:
                continue 
            if ratio_i < _MIN_MID_RATIO:
                continue

            Nclm_i  = dnsty_i * Npix_i * ratio_i
            
            print( f"Density: {dnsty_i}, Ratio: {ratio_i}" )
            if _VERBOSE: 
                print( f"Block mask of {Npix_i} points!" )
            # self.jps.arr_show( clstr )
            if Npix_i < self._CLUST_MIN:
                break # We sorted clusters descending
            # Test 2: Reasonable distance
            count_i  = np.count_nonzero( depArr[clstr] )
            if count_i == 0:
                continue
            depMsk_i = depArr[clstr].sum() / count_i
            if _VERBOSE: 
                print( f"Block mask is {depMsk_i} away!" )
            if (depMsk_i < self._DIST_MIN) or (depMsk_i > self._DIST_MAX):
                continue
            
            # Test 3: Expected size
            bbox_i = get_nonzero_mask_bbox( clstr )
            # span_i = [bbox_i[2]-bbox_i[0], bbox_i[3]-bbox_i[1],]
            span_i = [bbox_i[1][0]-bbox_i[0][0], bbox_i[1][1]-bbox_i[0][1],]
            if _VERBOSE: 
                print( f"Span is {span_i}" )
            angl_i = [ np.deg2rad( (span_i[0]/imgArr.shape[0])*env_var("_D405_FOV_V_DEG") ),
                       np.deg2rad( (span_i[1]/imgArr.shape[1])*env_var("_D405_FOV_H_DEG") ), ] 
            if _VERBOSE: 
                print( f"Arc is {angl_i}" )
            dims_i = [ 2.0 * np.tan( angl_i[0]/2.0 ) * depMsk_i, 
                       2.0 * np.tan( angl_i[1]/2.0 ) * depMsk_i, ] 
            scal_i = np.array( dims_i ) / env_var("_BLOCK_SCALE")
            # DANGER: HACK
            # factor = 1.0
            factor = list()
            for scl in scal_i:
                if scl < 1.0:
                    # factor *= scl
                    factor.append( scl )
                elif scl > 1.0:
                    # factor *= (1.0 - (scl - 1.0))
                    factor.append( 1.0 - (scl - 1.0) )
            
            # Nclm_i *= max( factor )  
            Nclm_i *= min( factor )  
            
            if _VERBOSE: 
                print( f"Scale is {scal_i} * {env_var('_BLOCK_SCALE')}" )
            if (self._SCAL_MIN <= scal_i[0] <= self._SCAL_MAX) and (self._SCAL_MIN <= scal_i[1] <= self._SCAL_MAX):
                # if Npix_i > pixMax:
                if Nclm_i > pixMax:
                    # pixMax = Npix_i
                    pixMax = Nclm_i
                    clstMx = clstr
        return clstMx  


    def dump_scene( self ):
        epPklPath = self.near_path( self.rFile ).replace( ".pkl", f"_{self.seq}.pkl" ) 
        if len( self.scenes ):
            currScene = self.scenes.popleft()
            with open( epPklPath, 'wb' ) as outFil:
                pickle.dump( currScene, outFil )
            self.seq += 1
            print( f"Saved: {epPklPath}!" )
        else:
            print( f"NO scene to save!" )    


    def new_scene( self ):
        """ Init Empty State Reconstruction """
        if len( self.current ):
            self.scenes.append( deepcopy( self.current ) )
        self.current = {
            "labels" : list(), #- List of objects in this scene
            "image"  : dict(), #- Lookup of color images used
            "depth"  : dict(), #- Lookup of depth images used
            "clouds" : deque(), # Collection of clouds obtained from the masked images
            "objects": deque(), # Collection of readings obtained from the masked images
            "sensed" : list(), # Collection of symbols obtained from the robot
            "symbols": dict(), #- Lookup of objects obtained from the readings
        }
        if len( self.scenes ) > self._N_RETAIN:
            self.dump_scene()


    def log_sensed( self, sensed ):
        """ Log what the robot saw """
        self.current['sensed'] = sensed


    def get_last_scene( self, backDex : int = 1 ) -> list[GraspObj]:
        """ Get Last State Reconstruction """
        if len( self.scenes ) >= backDex:
            # return deepcopy( self.scenes[-backDex] )
            return self.scenes[-backDex]
        else:
            return None


    def process_observation_data( self, obsData : dict, goalLabels : list[str], camPose : np.ndarray = None ):
        """ Transform observation data into information about the current state """
        if camPose is None:
            camPose = np.eye(4)
        inpt : dict       = obsData['input']
        iKey : str        = choice( list( inpt.keys() ) )
        imag : np.ndarray = inpt[ iKey ]['image']
        dpth : np.ndarray = inpt[ iKey ]['depth']

        # self.current['image'][ iKey ] = deepcopy( imag )
        # self.current['depth'][ iKey ] = deepcopy( dpth )
        self.current['image'][ iKey ] = imag
        self.current['depth'][ iKey ] = dpth
        
        Nadd = 0
        for label in goalLabels:
            res = self.find_block_mask( label, imag, dpth )
            if res is not None:
                self.current['labels'].append( label )
                print( f"MASK FOUND for {label}!" )
                # self.jps.arr_show( res )
                pcd_i = color_depth_to_pointcloud( imag, dpth, _DEPTH_MATX_1280x720, mask = res )
                transform_mpcd( pcd_i, camPose )
                pos_i = get_mpcd_pose( pcd_i )

                # self.current['clouds'].append( deepcopy( pcd_i ) )
                self.current['clouds'].append( pcd_i )
                
                obj_i = GraspObj( 
                    label = label, 
                    pose  = ObjPose( pos_i ), 
                    ts    = now(), 
                    score = 0.0,
                    # cpcd  = deepcopy( pcd_i ),
                    cpcd  = pcd_i,
                )
                if p_symbol_inside_workspace_bounds( 
                    obj_i, 
                    noPad     = False, 
                    addMargin = 0.120 # 0.060 # 0.120 
                ) and (len( pcd_i ) >= self._PCD_MIN): # Sometimes extraneous shit gets picked up!
                    if not _RAM_SAVER:
                        self.jps.arr_show( res )
                    self.current['objects'].append( obj_i )
                    Nadd += 1
                    print( f"{label} can be found at {obj_i}" )
                else:
                    print( f"{label} reading is OUT OF BOUNDS!, {obj_i}" )

        imag = None
        dpth = None

        print( f"\nAdded {Nadd} readings!\n\n" )


    @staticmethod
    def get_block_facts( objLst : list[GraspObj] ):
        """ Scan the environment for evidence that the task is progressing, using current beliefs """
        rtnFct = list()
        ## Ground the Blocks ##
        for sym in objLst:
            rtnFct.append( ('Graspable', sym.label,) )
            rtnFct.append( ('GraspObj' , sym.label, sym.pose, ) )
        ## Support Predicates && Blocked Status ##
        # Check if `sym_i` is supported by `sym_j`, blocking `sym_j`, NOTE: Table supports not checked
        # _XY_FACTOR = 1.125
        # _XY_FACTOR = 1.250
        # _XY_FACTOR = 1.500
        _XY_FACTOR = 1.750
        _Z_FACTOR = 0.750
        supDices = set([])
        for i, sym_i in enumerate( objLst ):
            for j, sym_j in enumerate( objLst ):
                if i != j:
                    lblUp = sym_i.label
                    lblDn = sym_j.label
                    posUp = extract_pose_as_homog( sym_i )
                    posDn = extract_pose_as_homog( sym_j )
                    xySep = diff_norm( posUp[0:2,3], posDn[0:2,3] )
                    zSep  = posUp[2,3] - posDn[2,3]
                    if ((xySep <= (env_var("_WIDE_XY_ACCEPT") * _XY_FACTOR)) and ( env_var("_WIDE_Z_ABOVE") * (1.0 + (1.0 - _Z_FACTOR)) >= zSep >= env_var("_SMUSH_Z_ABOVE") * _Z_FACTOR )):
                        supDices.add(i)
                        rtnFct.extend([
                            ('Supported', lblUp, lblDn,),
                            ('Blocked', lblDn,),
                            ('PoseAbove', ObjPose( posUp ), lblDn,),
                        ])
        for i, sym_i in enumerate( objLst ):
            if i not in supDices:
                pose_i = extract_pose_as_homog( sym_i )
                hght_i = pose_i[2,3]
                if env_var("_DEFAULT_TABLE_SUPPORT") or (hght_i <= env_var("_TABLE_SUPPORT_Z_MAX")):
                    rtnFct.extend( [
                        ('Supported', sym_i.label, 'table',),
                        ('PoseAbove', sym_i.pose , 'table',),
                    ] )
                else:
                    raise ValueError( f"FLOATING BLOCK: {sym_i}, {hght_i}" )
        ## Return relevant predicates ##
        return rtnFct


    @staticmethod
    def logical_Z_snap( objLst : list[GraspObj] ):
        """ Impoze zome phyzical rulez on the Z-coordinatez of the objectz """
        blockFacts = OCV_State_Tracker.get_block_facts( objLst )

        def get_obj_by_label( q : str ):
            """ Get the object that matches the label """
            nonlocal objLst
            for sym in objLst:
                if sym.label == q:
                    return sym
            return None
        
        def get_facts_by_type( typNam : str ):
            """ Get all the facts that match the type """
            nonlocal blockFacts
            rtnFct = list()
            for fact in blockFacts:
                if fact[0] == typNam:
                    rtnFct.append( fact )
            return rtnFct

        ## Stack the Blocks ##
        frontier = deque(["table",])
        modSet   = set([])
        while len( frontier ):  
            support = frontier.pop()
            facts_i = [item for item in get_facts_by_type( 'Supported' ) if (item[-1] == support)]
            if support == "table":
                for fct_j in facts_i:
                    obj_j = get_obj_by_label( fct_j[1] )
                    if obj_j is not None:
                        obj_j.pose.pose[2,3] = env_var("_BLOCK_SCALE")/2.0
                        modSet.add( obj_j.index )
                        frontier.append( obj_j.label )
            else:
                obj_b = get_obj_by_label( support )
                if obj_b is not None:
                    for fct_j in facts_i:
                        obj_j = get_obj_by_label( fct_j[1] )
                        obj_j.pose.pose[2,3] = obj_b.pose.pose[2,3] + env_var("_BLOCK_SCALE")
                        modSet.add( obj_j.index )
                        frontier.append( obj_j.label )
        for sym in objLst:
            if sym.index not in modSet:
                obj_j.pose.pose[2,3] = snap_z_to_nearest_block_unit_above_zero( obj_j.pose.pose[2,3] )


    def reconcile_scene( self ):
        """ Merge all the readings """
        _BLEND_FACTOR = 0.200 # 0.200 # 0.500

        def p_collides( reading : GraspObj, objDct : dict[str,GraspObj] ):
            for obj in objDct.values():
                if (euclidean_distance_between_symbols( reading, obj ) < env_var("_ACCEPT_POSN_ERR")) and (obj.label != reading.label):
                    return True
            return False

        for obj_i in self.current['objects']:
            if not p_symbol_inside_workspace_bounds( obj_i, noPad = True, addMargin = 0.120 ):
                continue
            if p_collides( obj_i, self.current['symbols'] ):
                continue
            # if len( obj_i.cpcd ):
            #     aabb = obj_i.cpcd.calc_aabb()
            #     if max(aabb[1,2], aabb[0,2]) < (0.25 * env_var("_BLOCK_SCALE")):
            #         continue
            lbl_i = obj_i.label
            # WARNING: THE FOLLOWING ASSUMES ONE OF EACH LABEL!
            if lbl_i in self.current['symbols']:
                obj_j  = self.current['symbols'][ lbl_i ]
                posn_i = extract_position( obj_i )
                posn_j = extract_position( obj_j )
                # dst_ij = np.linalg.norm( np.subtract( posn_i, posn_j ) )
                # factor = np.exp( -dst_ij*10 )*_BLEND_FACTOR
                # posn_r = posn_i * factor + posn_j * (1.0 - factor)
                posn_r = posn_i * _BLEND_FACTOR + posn_j * (1.0 - _BLEND_FACTOR)
                pose_r = extract_pose_as_homog( obj_j )
                pose_r[:3,3] = posn_r
                obj_j.pose = ObjPose( pose_r )
            else:
                # self.current['symbols'][ lbl_i ] = obj_i.copy()
                self.current['symbols'][ lbl_i ] = obj_i
        print( f"Processed {len(self.current['objects'])} readings!" )

        lstScn = self.get_last_scene()
        lstSet = set( lstScn['symbols'].keys() ) 
        curSet = set( self.current['symbols'].keys() )
        difSet = lstSet - curSet 
        for dLabel in difSet:
            # self.current['symbols'][ dLabel ] = deepcopy( lstScn['symbols'][ dLabel ] )
            self.current['symbols'][ dLabel ] = lstScn['symbols'][ dLabel ]

        rtnSym = list( self.current['symbols'].values() )
        
        self.logical_Z_snap( rtnSym )
        return rtnSym


    def near_path( self, parentPath : str, suffix : str = "_OCV-State", EXT : str = "pkl" ):
        """ A path similar to parent path, but with a suffix """
        return parentPath.split('.')[0] + suffix + "." + EXT
    

    def current_scene_confusion( self, sensedObjects : list[GraspObj] ):
        """ Return the number of `sensedObjects` that *contradict* the current scene """
        lastScen = list( self.current['symbols'].values() )
        matches  = dict()
        
        if (sensedObjects is None) or (not len( sensedObjects )):
            return {
            "N_total"  : len( lastScen ),
            "N_confuse": 0,
            "N_halluc" : 0,
            "N_missing": len( lastScen ),
            "N_sensed" : 0,
            "N_true"   : len( lastScen ),
        }
        
        # lastScen = self.get_last_scene()
        

        # Store Sensed Objects #
        for obj_i in sensedObjects:
            matches[ id( obj_i ) ] = { "sensed" : obj_i, "known" : None, "d" : 6e10 }
        
        # Match OpenCV Objects to Sensed Objects #
        for obj_j in lastScen:
            dMin   = 6e10
            kMin_i = None
            for k_i, v_i in matches.items():
                d_ij = euclidean_distance_between_symbols( v_i["sensed"], obj_j )
                if d_ij is None:
                    continue
                if d_ij <= env_var("_ACCEPT_POSN_ERR") and d_ij < dMin:
                    dMin   = d_ij
                    kMin_i = k_i 
            if kMin_i is not None:
                matches[ kMin_i ]["known"] = obj_j
                matches[ kMin_i ]["d"    ] = dMin
            
        # Compute Number of Total, Confused, Hallucinated, and Missing Objects #
        Ntot = max( len( lastScen ), len( sensedObjects ) ) # Total number of objects
        Ncnf = 0 # Number of confusions
        Nhal = 0 # Number of hallucinations, False positive
        for k_i, v_i in matches.items():
            # If the sensed block was real, Then check for confusion
            if v_i["known"] is not None:
                if v_i["known"].label != v_i["sensed"].label:
                    Ncnf += 1
            # Else block was NOT real, The system hallucinated it! 
            else:
                Nhal += 1 
        Nmis = Ntot - len( sensedObjects )
        return {
            "N_total"  : Ntot,
            "N_confuse": Ncnf,
            "N_halluc" : Nhal,
            "N_missing": Nmis,
            "N_sensed" : len( sensedObjects ),
            "N_true"   : len( lastScen ),
        }
    

    @staticmethod
    def action_2_move( action : dict[str,list[str]] = None ) -> dict[str,np.ndarray]:
        """ Express the action as a move from one pose to another """
        if action is None:
            return None
        if 'next' in action:
            seq = action['next']
        elif 'plan' in action:
            seq = action['plan'][:4]
        else:
            return None
        src = None
        dst = None
        for bhv in seq:
            if "Pick" in bhv[:10]:
                src = extract_pose_from_str( bhv )
            elif ("Place" in bhv[:10]) or ("Stack" in bhv[:10]):
                dst = extract_pose_from_str( bhv )
        if (src is not None) and (dst is not None):
            return {"src" : src, "dst" : dst,}
        else:
            return None


    def ingest_action( self, action : dict[str,list[str]] = None ):
        """ Log plan for eval later """
        moveDict = None
        if (action is not None) and len( action ):
            moveDict = OCV_State_Tracker.action_2_move( action )
        self.actions.append( moveDict )


    def check_move_outcome( self, move : dict[str,np.ndarray] ):
        """ Was there a transition that matches these poses?? """
        # ASSUMPTION: IF WE FIND OBJECTS AT THE SOURCE AND THE DESTINATION, THEN THE MOVE OCCURRED AS PLANNED
        if len( self.scenes ) >= 2:
            lastScen = self.get_last_scene(1)
            prevScen = self.get_last_scene(2)
            # Find the Source Object #
            srcObj = None
            srcErr = 6e10
            for obj_p in prevScen:
                d = euclidean_distance_between_symbols( move['src'], obj_p )
                if d <= env_var("_ACCEPT_POSN_ERR") and d < srcErr:
                    srcErr = d
                    srcObj = obj_p 
            # Find the Destination Object #
            dstObj = None
            dstErr = 6e10
            for obj_l in lastScen:
                d = euclidean_distance_between_symbols( move['dst'], obj_l )
                if d <= env_var("_ACCEPT_POSN_ERR") and d < dstErr:
                    dstErr = d
                    dstObj = obj_l
            # Did we find both? #
            return (srcObj is not None) and (dstObj is not None)
        else:
            return None


    def dump_episode( self, epPklPath : str, nameSimilar : bool = True ):
        """ Write a file that describes the results of each step of the episode """
        if nameSimilar:
            epPklPath = self.near_path( epPklPath )
        self.new_scene() # Save last scene
        while len( self.scenes ):
            self.dump_scene()
        self.scenes = deque()
        # print( f"Saved: {epPklPath}!" )