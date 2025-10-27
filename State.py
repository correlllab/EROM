import os, pickle, time, subprocess, math
now = time.time 
from collections import deque, defaultdict
from datetime import datetime
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as patches

import numpy as np

from utils import deep_copy_memory_list, snap_z_to_nearest_block_unit_above_zero
from aspire.env_config import env_var
from aspire.symbols import ( ObjPose, GraspObj, euclidean_distance_between_symbols, extract_pose_as_homog )

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
    pass


def blu_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Blue Block """
    pass


def grn_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Green Block """
    pass


def blk_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the Black Block """
    pass


def wht_block_mask( img : np.ndarray ) -> np.ndarray:
    """ Return a mask that segments the White Block """
    pass



##### "Ground Truth" Tracker ############################################## 

class OCV_State_Tracker:
    """ Use OpenCV to infer something closer to the "Ground Truth", Prefer plain JSON """
    def __init__( self ):
        """ Set up tracking """
        self.scenes  = deque() # Sequence of reconstruction data
        self.states  = deque() # Sequence of States
        self.changes = deque() # Sequence of Transitions
        self.current = dict() #- All data relating to the current state

    def new_scene( self ):
        """ Init Empty State Reconstruction """
        if len( self.current ):
            self.scenes.append( deepcopy( self.current ) )
        self.current = {
            "???": None,
        }


    def mask_to_ray( self ):
        # FIXME: IS THERE RAY DATA ALREADY THERE?????????? COMPARE ???????
        pass


    def process_observation( self, obsData, goalLabels : list[str] ):
        """ Transform observation data into information about the current state """
        pass


    def infer_current_state( self ) -> list[GraspObj]:
        """ Return a list of blocks, classes, and poses """
        pass


    def compare_states( self, ocvState : list[GraspObj], eromState : list[GraspObj] ):
        """ Return a list of object-wise differences between the two states """
        pass


    def episode_report( self, epPklPath : str ):
        """ Write a file that describes the results of each step of the episode, Prefer plain JSON """
        pass


    def diagnose_diff( self ):
        # FIXME: TRY TO DISCERN WHY THE DIFFERENCES OCCURRED????
        pass
