########## INIT ####################################################################################

import time, os
now = time.time 
from math import isnan
from collections import deque
from datetime import datetime
from uuid import uuid4

import numpy as np
np.set_printoptions( precision = 4 )
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt


from aspire.symbols import GraspObj
from aspire.env_config import env_var



########## HELPER FUNCTIONS ########################################################################

def zip_dict_sorted_by_decreasing_value( dct ):
    """ Return a list of (k,v) tuples sorted by decreasing value """
    keys = list()
    vals = list()
    for k, v in dct.items():
        keys.append(k)
        vals.append(v)
    return sorted( zip( keys, vals ), key=lambda x: x[1], reverse=1)



########## MEMORY FUNCTIONS ########################################################################

def copy_as_LKG( sym : GraspObj ):
    """ Make a copy of this belief for the Last-Known-Good collection """
    rtnObj = sym.copy()
    rtnObj.LKG = True
    return rtnObj


def copy_readings_as_LKG( readLst ):
    """ Return a list of readings intended for the Last-Known-Good collection """
    rtnLst = list()
    for r in readLst:
        rtnLst.append( copy_as_LKG( r ) )
    return rtnLst


def mark_readings_LKG( readLst : list[GraspObj], val : bool = True ):
    """ Return a list of readings intended for the Last-Known-Good collection """
    for r in readLst:
        r.LKG = val


def entropy_factor( probs ):
    """ Return a version of Shannon entropy scaled to [0,1] """
    if isinstance( probs, dict ):
        probs = list( probs.values() )
    tot = 0.0
    # N   = 0
    for p in probs:
        pPos = max( p, 0.00001 )
        tot -= pPos * np.log( pPos )
            # N   += 1
    return tot / np.log( len( probs ) )


def set_quality_score( obj : GraspObj ):
    """ Calc the score for this `GraspObj` """
    score_i = (1.0 - entropy_factor( obj.labels )) * obj.count
    if isnan( score_i ):
        print( f"\nWARN: Got a NaN score with count {obj.count} and distribution {obj.labels}\n" )
        score_i = 0.0
    obj.score = score_i


def snap_z_to_nearest_block_unit_above_zero( z : float ):
    """ SNAP TO NEAREST BLOCK UNIT && SNAP ABOVE TABLE """
    sHalf = (env_var("_BLOCK_SCALE")/2.0)
    zBump = sHalf + env_var("_Z_TABLE")
    zUnit = np.rint( (z-zBump+env_var("_Z_SNAP_BOOST")) / env_var("_BLOCK_SCALE") ) # Quantize to multiple of block unit length
    zBloc = max( (zUnit*env_var("_BLOCK_SCALE"))+zBump, zBump )
    return zBloc


def get_pose_attr( target ):
    """ Dig out `pose` by name """
    while hasattr( target, 'pose' ):
        target = target.pose
    return np.array( target )


def deep_copy_memory_list( mem : list[GraspObj] ):
    """ Make a deep copy of the memory list """
    rtnLst = list()
    for m in mem:
        rtnLst.append( m.copy() )
    return rtnLst



########## GEOMETRY FUNCTIONS ######################################################################

def closest_ray_points( A_org, A_dir, B_org, B_dir ):
    """ Return the closest point on ray A to ray B and on ray B to ray A """
    # https://palitri.com/vault/stuff/maths/Rays%20closest%20point.pdf
    c  = np.subtract( B_org, A_org )
    aa = np.dot( A_dir, A_dir )
    bb = np.dot( B_dir, B_dir )
    ab = np.dot( A_dir, B_dir )
    ac = np.dot( A_dir, c     )
    bc = np.dot( B_dir, c     ) 
    dv = (aa*bb-ab*ab)
    if dv < 0.0001:
        raise ValueError( f"BAD DIVISOR computed from: \n{A_org=}, {A_dir=}, \n{B_org=}, {B_dir=}" )
    fA = (-ab*bc + ac*bb) / dv
    fB = ( ab*ac - bc*aa) / dv
    pA = np.add( A_org, np.multiply( A_dir, fA ) )
    pB = np.add( B_org, np.multiply( B_dir, fB ) )
    return pA, pB



########## HELPER CLASSES ##########################################################################

class JupyterPlotServer:
    """ New Plots Forever """
    def __init__( self, imgDir : str = "data/MemImg" ):
        """ Init vars to save plots as PNG in the BG """
        self.imgDir   = imgDir
        self.imgEXT   = "png"
        self.figure   = None
        self.bboxMode = 'tight'
        self.dfltSize = (8,6,)


    def fig( self, figsize = None ):
        """ New Figure """
        if figsize is None:
            figsize = self.dfltSize
        self.figure = plt.figure( figsize = figsize )


    def img_show( self, path ):
        """ Load Image, Show Image, Close Image """
        # Load the image
        img = Image.open( path )
        # Display the image
        display( img )
        # Close the image
        img.close()
        # Erase the image
        img = None


    def plt_show( self ):
        """ Save the current figure as an image, Close the figure, Display the image """
        figPath = os.path.join( self.imgDir, f"{uuid4()}.{self.imgEXT}" )
        plt.savefig( figPath, bbox_inches = self.bboxMode )
        plt.close( self.figure )
        self.img_show( figPath )


    def arr_show( self, arr, noAxes = True ):
        """ Display an array """
        self.fig()
        if noAxes:
            # Turn off the axes
            plt.axis('off')
        plt.imshow( arr )
        self.plt_show()



########## ANALYSIS FUNCTIONS ######################################################################

def print_header( text : str, preWidth : int, totWidth : int, capitalize = True, _HDR_CHR : str = '#' ):
    """ Print a pleasant header """
    if capitalize:
        text = f"{text}".upper()
    totStr = '\n'*int(totWidth/25) + f"{preWidth*_HDR_CHR[0]} {text} "
    pstStr = max( totWidth-len(totStr)+1, 0 )*_HDR_CHR[0]
    if not len( pstStr ):
        pstStr = f"{preWidth*_HDR_CHR[0]}"
    totStr += pstStr
    print( totStr )


def dex_key( x, offset = -1 ):
    dex = f"{x}".split('_')[ offset ].replace( ".pkl", "" )
    if len( dex ) >= 2:
        return dex
    elif len( dex ) < 2:
        return '0'*(2-len( dex )) + dex
    else:
        raise ValueError( "`dex_key`: This should NOT have happened!" )


def play_tone( duration_s = 5, freq_Hz = 650 ):
    """ Play a notification tone """
    os.system( f'play -nq -t alsa synth {duration_s} sine {freq_Hz}' )


def crash_out( notify = True ):
    """ End the program with Brutal Finality """
    if notify:
        play_tone()
    print( "\n\n" )
    os.system( 'kill %d' % os.getpid() ) 



########## PLOTTING FUNCTIONS ######################################################################
_TITLE_FONT_SIZE =  13
_TIGHT_MARGIN    =   0.05
_DEFAULT_DIV     = 100 #80 #100 #200


def xy_plot_filled_under( X, Y, plotTitle = None, fName = "output.pdf", xLabel = None, yLabel = None, 
                          titleFontSize_pt = _TITLE_FONT_SIZE ):
    """ Creat cumulative curve """
    # Plot line
    plt.plot( X, Y )

    # Shade the area under the curve
    plt.fill_between( X, Y, 0, color = 'skyblue', alpha = 0.5 )

    if plotTitle is not None:
        plt.title( plotTitle, fontsize = titleFontSize_pt ) # Set the title && font size
    if xLabel is not None:
        plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    if yLabel is not None:
        plt.ylabel( yLabel ) # ---------------- Setting the y-axis label

    plt.show()


def make_histo( series, plotTitle, xLabel = 'Makespan', yLabel = 'Occurrences', savefig = True ):
    """ Create Histogram """
    if savefig:
        plt.clf()
    plt.margins( _TIGHT_MARGIN )
    print( f"\n{plotTitle}" )
    print( f"Mean: ___ {np.mean(series)}" )
    print( f"Median: _ {np.median(series)}" )
    print( f"Std.Dev.: {np.std(series)}" )
    plt.hist( series, _DEFAULT_DIV )
    plt.title( plotTitle, fontsize = _TITLE_FONT_SIZE ) # Set the title && font size
    plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    plt.tight_layout()
    plt.show()



########## PARSING #################################################################################

def tokenize( expr : str ):
    """ Break a text `expr` into parts """
    _reserved = ['[', ']', ',',]
    expr += ' ' # Terminator hack
    token  = ""
    tokens = deque()

    def p_reserved( char ):
        return (char in _reserved)

    def store_token():
        nonlocal token, tokens
        if len( token ):
            try:
                tokens.append( float( token ) )
            except ValueError:
                tokens.append( token )
        token  = ""

    def store_char( char ):
        nonlocal tokens
        store_token()
        tokens.append( char )

    for char in expr:
        if char.isspace():
            store_token()
        elif p_reserved( char ):
            store_char( char )
        else:
            token += char

    return tokens


def extract_pose_from_tokens( tokens : list[str] ):
    """ Tokenize and parse a pose string """
    depth = 0
    matrx = deque()
    array = deque()
    for token in tokens:
        if token =='[':
            depth += 1
        if token ==']':
            depth -= 1
        if (depth == 2) and (not isinstance( token, str )):
            array.append( token )
        elif depth == 1:
            if len( array ):
                matrx.append( list( array ) )
                array = deque()
    if depth == 0:
        return np.array( list( matrx ) )
    else:
        return None
    

def extract_name_from_tokens( tokens : list[str] ):
    """ Get a block name from a list of tokens """
    for token in tokens:
        if 'Block' in token:
            return token
    return None


def get_name_and_origin( lines : list[str] ) -> np.ndarray:
    """ Get the origin and name of the block """
    accum  = False
    tokens = deque()
    name   = None
    pose   = None
    for line in lines:
        if ('Pick' in line) or ('Unstack' in line):
            accum = True
        if accum:
            linTkn = tokenize( line )
            tokens.extend( linTkn )
            name = extract_name_from_tokens( tokens )
            pose = extract_pose_from_tokens( tokens )
            if (name is not None) and (pose is not None):
                return name, pose
    return None, None


def get_desination( lines : list[str] ) -> np.ndarray:
    """ Get the origin and name of the block """
    accum  = False
    tokens = deque()
    pose   = None
    for line in lines:
        if ('Place' in line) or ('Stack' in line):
            accum = True
        if accum:
            linTkn = tokenize( line )
            tokens.extend( linTkn )
            pose = extract_pose_from_tokens( tokens )
            if (pose is not None):
                return pose
    return None


def parse_action( action : dict[str,list[str]] = None ):
    """ Get the intended class, origin, destination of the block """
    if action is not None:
        Lines    = action['next']
        dst      = get_desination( Lines )
        nam, src = get_name_and_origin( Lines )
    return {
        'name'   : nam,
        'bgnPose': src,
        'endPose': dst,
    }
