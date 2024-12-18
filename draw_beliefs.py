########## INIT ####################################################################################
from pprint import pprint

import numpy as np

import vispy
from vispy import scene
from vispy import gloo
from vispy.visuals import transforms
from vispy.color import Color

import numpy as np

### Local ###
from aspire.env_config import env_var, env_sto
from aspire.homog_utils import posn_from_xform, vec_unit, homog_xform
from aspire.symbols import extract_pose_as_homog, GraspObj

_TABLE_THIC = 0.015




########## ENVIRONMENT #############################################################################

def set_render_env():
    """ Set vars used to draw EROM memories """
    env_sto( "_SCAN_ALPHA", 0.5 )



########## HELPER FUNCTIONS ########################################################################

def zip_dict_sorted_by_decreasing_value( dct ):
    """ Return a list of (k,v) tuples sorted by decreasing value """
    keys = list()
    vals = list()
    for k, v in dct.items():
        keys.append(k)
        vals.append(v)
    return sorted( zip( keys, vals ), key=lambda x: x[1], reverse=1)


def look_at_matrix( target, eye, up = None ):
    """ Construct the camera transformation """
    if up is None:
        up = [0,0,1,]
    else:
        up = vec_unit( up )
    zBasis = vec_unit( np.subtract( target, eye ) )
    xBasis = vec_unit( np.cross( up, zBasis ) )
    yBasis = np.cross( zBasis, xBasis )
    rtnMtx = np.eye(4)
    rtnMtx[0:3,0] = xBasis
    rtnMtx[0:3,1] = yBasis
    rtnMtx[0:3,2] = zBasis
    rtnMtx[0:3,3] = eye
    return np.transpose( rtnMtx )



########## DISPLAY WINDOW ##########################################################################

def vispy_geo_list_window( geoLst ):
    canvas = scene.SceneCanvas( keys='interactive', size=(1000, 900), show=True )
    # Enable backface culling
    gloo.set_state( cull_face = True )
    # vispy.gloo.wrappers.set_state( cull_face = True )
    # vispy.gloo.wrappers.set_cull_face( mode = 'back' )

    # Set up a viewbox to display the cube with interactive arcball
    view = canvas.central_widget.add_view()
    view.bgcolor = '#ffffff'

    view.camera = scene.ArcballCamera() #'arcball'
    view.camera.up = np.array( [0.0, 0.0, 1.0] )
    view.camera.center = np.array([
        env_var("_MIN_X_OFFSET") + env_var("_X_WRK_SPAN")/2.0, 
        env_var("_MIN_Y_OFFSET") + env_var("_Y_WRK_SPAN")/2.0, 
        0.0,
    ])
    view.camera.distance = 0.75

    view.padding = 100
    view.add( scene.visuals.XYZAxis() )

    for geo in geoLst:
        view.add( geo )
    

    canvas.app.run()



########## DRAWING FUNCTIONS #######################################################################

def table_geo():
    """ Draw the usable workspace """
    # table  = o3d.geometry.TriangleMesh.create_box( _X_WRK_SPAN, _Y_WRK_SPAN, _TABLE_THIC )
    table  = scene.visuals.Box( env_var('_X_WRK_SPAN'), _TABLE_THIC, env_var("_Y_WRK_SPAN"),  
                                color = [237/255.0, 139/255.0, 47/255.0, 1.0], edge_color="black" , )
    table.transform = transforms.STTransform( translate = (
        env_var("_MIN_X_OFFSET") + env_var("_X_WRK_SPAN")/2.0, 
        env_var("_MIN_Y_OFFSET") + env_var("_Y_WRK_SPAN")/2.0, 
        -_TABLE_THIC/2.0
    ) )
    return table


def wireframe_box_geo( xScl, yScl, zScl, color = None ):
    """ Draw a wireframe cuboid """
    if color is None:
        color = [0,0,0,1]
    xHf = xScl/2.0
    yHf = yScl/2.0
    zHf = zScl/2.0
    verts = np.array([
        [ -xHf, -yHf, +zHf ], # 0
        [ +xHf, -yHf, +zHf ], # 1
        [ +xHf, +yHf, +zHf ], # 2
        [ -xHf, +yHf, +zHf ], # 3
        [ -xHf, -yHf, -zHf ], # 4
        [ +xHf, -yHf, -zHf ], # 5
        [ +xHf, +yHf, -zHf ], # 6
        [ -xHf, +yHf, -zHf ], # 7   
    ])
    ndces = np.array([
        [0,4,],
        [4,5,],
        [5,6,],
        [6,7,],

        [0,1,],
        [1,2,],
        [2,3,],
        [3,0,],

        [7,4,],
        [1,5,],
        [2,6,],
        [3,7,],
    ])
    wireBox = scene.visuals.Line(
        pos     = verts,
        connect = ndces,
        color   = color,
    )
    return wireBox


def wireframe_box_neg( xScl, yScl, zScl, color = None ):
    """ Draw a wireframe cuboid """
    if color is None:
        color = [0,0,0,1]
    xHf = xScl/2.0
    yHf = yScl/2.0
    zHf = zScl/2.0
    verts = np.array([
        [ -xHf, -yHf, +zHf ], # 0
        [ +xHf, -yHf, +zHf ], # 1
        [ +xHf, +yHf, +zHf ], # 2
        [ -xHf, +yHf, +zHf ], # 3
        [ -xHf, -yHf, -zHf ], # 4
        [ +xHf, -yHf, -zHf ], # 5
        [ +xHf, +yHf, -zHf ], # 6
        [ -xHf, +yHf, -zHf ], # 7   
    ])
    ndces = np.array([
        
        [4,5,], # Cycle 1
        [5,6,],
        [6,7,],
        [7,4,],

        [1,2,], # Cycle 2
        [2,3,],
        [3,0,],
        [0,1,],
        
        [0,4,], # Pillars
        [1,5,],
        [2,6,],
        [3,7,],

        [4,6,], # Cross-beams
        [5,7,],
        [1,3,],
        [2,0,],
        [4,3,],
        [4,1,],
        [5,0,],
        [5,2,],
        [6,3,],
        [6,1,],
        [7,0,],
        [7,2,],
    ])
    wireBox = scene.visuals.Line(
        pos     = verts,
        connect = ndces,
        color   = color,
    )
    return wireBox


def quality_bar( objReading : GraspObj, maxScl = None, maxLen = None ):
    pass


def reading_geo( objReading : GraspObj, alpha = None ):
    """ Get geo for a single observation """
    belClr = [0.5, 0.0, 1.0, 1.0,]
    lkgClr = [1.0, 0.0, 0.0, 1.0,]
    labelSort = zip_dict_sorted_by_decreasing_value( objReading.labels )
    objXfrm   = extract_pose_as_homog( objReading, noRot = True )
    # posn      = objPose[:3]
    # ornt      = objPose[3:]
    hf        = env_var("_BLOCK_SCALE")/2.0
    topCrnrs  = [
        homog_xform( np.eye(3), [-hf+hf,-hf+hf, env_var("_BLOCK_SCALE"),] ),
        homog_xform( np.eye(3), [-hf+hf, hf+hf, env_var("_BLOCK_SCALE"),] ),
        homog_xform( np.eye(3), [ hf+hf,-hf+hf, env_var("_BLOCK_SCALE"),] ),
        homog_xform( np.eye(3), [ hf+hf, hf+hf, env_var("_BLOCK_SCALE"),] ),
    ]
    rtnGeo  = list()
    clr = lkgClr if objReading.LKG else belClr 
    clr[-1] = env_var("_BLOCK_ALPHA") if (alpha is None) else alpha
    wir = wireframe_box_geo( env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), 
                             color = clr )
    wir.transform = transforms.STTransform( translate = objXfrm[:3,3] )
    rtnGeo.extend( [wir,] )
    
    for i in range(3):
        objXfrm[i,3] -= hf
    for i in range( 0, min( len(labelSort), len(topCrnrs) ) ):
        prob_i = labelSort[i][1]
        if (prob_i > 0.0):
            scal_i  = env_var("_BLOCK_SCALE") * prob_i
            xfrm_i = topCrnrs[i-1]
            # for j in range(3):
            #     xfrm_i[j,3] -= scal_i/2.0
            xfrm_i = objXfrm.dot( xfrm_i )

            # pprint( env_var("_CLR_TABLE") )
            colr_i = env_var("_CLR_TABLE")[ labelSort[i][0][:3] ]
            colr_i.append( 1.0 )
            colr_i[-1] = env_var("_BLOCK_ALPHA") if (alpha is None) else alpha
            bloc_i = scene.visuals.Box( scal_i, scal_i, scal_i,  
                                        color = colr_i, edge_color = lkgClr if objReading.LKG else belClr , )
            bloc_i.transform = transforms.STTransform( translate = xfrm_i[:3,3] )
            rtnGeo.append( bloc_i )
    if objReading.prob > 0.0:
        scl  = env_var("_BLOCK_SCALE") * objReading.prob
        bClr = env_var("_CLR_TABLE")[ objReading.label[:3] ]
        bClr.append( 1.0 )
        bClr[-1] = env_var("_BLOCK_ALPHA") if (alpha is None) else alpha
        blc  = scene.visuals.Box( scl, scl, scl,  
                                  color = bClr, edge_color="black" )
        for i in range(3):
            # objXfrm[i,3] += hf-(scl/2.0)
            pass
        blc.transform = transforms.STTransform( translate = objXfrm[:3,3] )
        rtnGeo.extend( [blc,] )
    return rtnGeo


def reading_list_geo( objs : list[GraspObj] ):
    """ Get geo for a list of observations """
    rtnGeo = [table_geo(),]
    for obj in objs:
        rtnGeo.extend( reading_geo( obj ) )
    return rtnGeo


def scan_list_geo( objs : list[GraspObj] ):
    """ Get geo for a list of observations """
    rtnGeo = [table_geo(),]
    for obj in objs:
        rtnGeo.extend( reading_geo( obj, alpha = env_var("_SCAN_ALPHA") ) )
    return rtnGeo


def symbol_geo( sym : GraspObj ):
    objXfrm = extract_pose_as_homog( sym, noRot = True )
    wf1 = wireframe_box_neg( env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), 
                             color = Color( "black" ) )
    wf1.transform = transforms.STTransform( translate = objXfrm[:3,3] )
    wf2 = wireframe_box_neg( env_var("_BLOCK_SCALE")*1.125, env_var("_BLOCK_SCALE")*1.125, env_var("_BLOCK_SCALE")*1.125, 
                             color = Color( "black" ) )
    wf2.transform = transforms.STTransform( translate = objXfrm[:3,3] )
    scl  = env_var("_BLOCK_SCALE") * 0.200
    bClr = env_var("_CLR_TABLE")[ sym.label[:3] ]
    bClr.append( env_var("_BLOCK_ALPHA") )
    blc  = scene.visuals.Box( scl, scl, scl,  
                              color = bClr, edge_color="black" )
    blc.transform = transforms.STTransform( translate = objXfrm[:3,3] )
    return [wf1, wf2, blc,] 


def symbol_neg( sym : GraspObj ):
    objXfrm = extract_pose_as_homog( sym, noRot = True )
    wf1 = wireframe_box_geo( env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), 
                             color = Color( "black" ) )
    wf1.transform = transforms.STTransform( translate = objXfrm[:3,3] )
    wf2 = wireframe_box_geo( env_var("_BLOCK_SCALE")*1.125, env_var("_BLOCK_SCALE")*1.125, env_var("_BLOCK_SCALE")*1.125, 
                             color = Color( "black" ) )
    wf2.transform = transforms.STTransform( translate = objXfrm[:3,3] )
    scl  = env_var("_BLOCK_SCALE") * sym.prob
    bClr = env_var("_CLR_TABLE")[ sym.label[:3] ]
    bClr.append( env_var("_BLOCK_ALPHA") )
    blc  = scene.visuals.Box( scl, scl, scl,  
                              color = bClr, edge_color="black" )
    blc.transform = transforms.STTransform( translate = objXfrm[:3,3] )
    return [wf1, wf2, blc,] 


def symbol_list_geo( objs : list[GraspObj], noTable = True ):
    """ Get geo for a list of symbols """
    if noTable:
        rtnGeo = list()
    else:
        rtnGeo = [table_geo(),]
    for obj in objs:
        rtnGeo.extend( symbol_geo( obj ) )
    return rtnGeo



########## RENDER MEMORY ###########################################################################

def render_memory_list( objs : list[GraspObj], syms = None ):
    """ Render the memory """
    objLst = reading_list_geo( objs )
    if syms is not None:
        objLst.extend( symbol_list_geo( syms, noTable = True ) )
    vispy_geo_list_window( objLst )



########## MAIN ####################################################################################
if __name__ == "__main__":
    from aspire.env_config import set_blocks_env
    set_blocks_env()

    neg = wireframe_box_neg( 0.050, 0.050, 0.050, color = "green" )
    vispy_geo_list_window( [neg,] )
    

    