import os, pickle, time, subprocess, math
now = time.time 
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as patches

import numpy as np

from utils import  deep_copy_memory_list
from aspire.env_config import env_var
from aspire.symbols import ( ObjPose, GraspObj, euclidean_distance_between_symbols )

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


    def visualize_last_segmentation( self ):
        """ Overlay all the camera shots with the segmentations """
        _SEG_TAG = 'ObsMeta' # 'meta'
        _DAT_DIR = 'data'
        NdataPts = len( self.log )
        # 0. Fetch the actual data
        datum = None
        for i in range( NdataPts ):
            datum_i = self.log[-i]
            if datum_i['msg'] == _SEG_TAG:
                if datum is None:
                    datum = datum_i
                else:
                    datum['data']['input'].update( datum_i['data']['input'] )
            elif (datum_i['msg'] != 'RobotState') and (datum is not None):
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


    def last_known_symbols( self ):
        """ Get last known symbol locations, even if we goofed last time """
        if len( self.symbols ):
            rtnSym = self.symbols[-1]
            index  = 2
            while ((not len( rtnSym )) and (index <= len( self.symbols ))):
                rtnSym = self.symbols[ -index ]
                index += 1
            return rtnSym
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
        self.symbols.append( deep_copy_memory_list( symLst ) )


    def log_beliefs( self, symLst ):
        """ Store the most recent symbols """
        self.beliefs.append( deep_copy_memory_list( symLst ) )


    def log_successful_action( self, poseBgn, poseEnd ):
        """ Move the symbol to where the robot moved it """
        self.trouble = False
        print( f"Moved block by {euclidean_distance_between_symbols( poseBgn, poseEnd )}" )

        lastFrame = deep_copy_memory_list( self.last_known_symbols() )

        if len( lastFrame ):
            dMin = 1e9
            sCls : GraspObj = None
            for sym in lastFrame:
                d = euclidean_distance_between_symbols( sym, poseBgn )
                if d < dMin:
                    dMin = d
                    sCls = sym
            sCls.pose = ObjPose( poseEnd )

            # lastFrame = self.repair_symbol_poses( lastFrame )

            self.symbols.append( lastFrame[:] )
        

    def log_failed_action( self, poseBgn, poseEnd ):
        """ We done goofed, Erase symbol """
        self.trouble = True
        self.symbols.append( list() )
        # if (len( self.beliefs ) > 1):
        #     self.beliefs.pop()
        print( f"Could NOT move block by {euclidean_distance_between_symbols( poseBgn, poseEnd )}" )


    # def repair_symbol_poses( self, symLst : list[GraspObj], maxDiff = None ):
    def repair_symbol_poses( self, symLst : list[GraspObj], maxDiff = None ) -> list[GraspObj]:
        """ Adjust the positions of symbols to their last """
        lastFrame : list[GraspObj] = self.last_known_symbols()
        rtnSym = list()
        lSet   = set([])
        cSet   = set([])
        dlta   = False

        def p_collide_return( qSym ):
            """ Did we already log a symbol at this location? """
            for rSym in rtnSym:
                # if euclidean_distance_between_symbols( qSym, rSym ) < env_var('_BLOCK_SCALE')*0.75:
                if euclidean_distance_between_symbols( qSym, rSym ) < env_var('_BLOCK_SCALE')*0.65:
                    return True
            return False

        if self.fixLabel and self.fixPose:
            for j, lSym in enumerate( lastFrame ):
                lSet.add( lSym.label )
                rtnSym.append( lSym )
            for i, rSym in enumerate( symLst ):
                if rSym.label not in lSet:
                    lSet.add( rSym.label )
                    rtnSym.append( rSym )
                    dlta = True

        elif self.fixPose:
            if maxDiff is None:
                maxDiff = 4.0 * env_var('_BLOCK_SCALE')

            print( "CHEAT OBJECTS:" )
            for j, lSym in enumerate( lastFrame ):
                print( f"\t{lSym}" )

            for i, rSym in enumerate( symLst ):
                sMin = None
                dMin = 1e9
                for j, lSym in enumerate( lastFrame ):
                    d_ij = euclidean_distance_between_symbols( rSym, lSym )
                    if (d_ij <= maxDiff) and (d_ij < dMin) and (not p_collide_return( lSym )):
                        dMin = d_ij
                        sMin = lSym
                if sMin is not None:
                    lSet.add( id( lSym ) )
                    cSet.add( rSym.label )
                    rSym.pose = sMin.pose
                    dlta = True
                if not p_collide_return( rSym ):
                    rtnSym.append( rSym )

            if env_var("_CHEAT_LKG"):
                for j, lSym in enumerate( lastFrame ):
                    if (lSym.label not in cSet) and (not p_collide_return( lSym )):
                        cSet.add( lSym.label )
                        rtnSym.append( lSym )
                
        if dlta:
            self.symbols.append( rtnSym )
        return rtnSym
        
