########## INIT ####################################################################################
from pprint import pprint

import numpy as np
from vispy import scene
from vispy.visuals import transforms
from vispy.color import Color
import numpy as np

### Local ###
from aspire.env_config import env_var
from aspire.homog_utils import posn_from_xform, vec_unit, homog_xform
from aspire.symbols import extract_pose_as_homog, GraspObj

_TABLE_THIC = 0.015



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
    # vispy.gloo.wrappers.set_state( cull_face = True )
    # vispy.gloo.wrappers.set_cull_face( mode = 'back' )

    # Set up a viewbox to display the cube with interactive arcball
    view = canvas.central_widget.add_view()
    view.bgcolor = '#ffffff'
    view.camera = 'arcball'
    view.padding = 100
    # view.camera.transform.matrix = look_at_matrix( target, eye )
    view.add( scene.visuals.XYZAxis() )

    for geo in geoLst:
        view.add( geo )
        # canvas.draw_visual( geo )
    
    # view.update()

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
        [1,5,],
        [2,6,],
        [3,7,],
        [0,1,],
        [1,2,],
        [2,3,],
        [3,0,],
        [4,5,],
        [5,6,],
        [6,7,],
        [7,4,],
    ])
    wireBox = scene.visuals.Line(
        pos     = verts,
        connect = ndces,
        color   = color,
    )
    return wireBox


def reading_geo( objReading : GraspObj ):
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
    
    wir = wireframe_box_geo( env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), 
                             color = lkgClr if objReading.LKG else belClr )
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
            colr_i.append( env_var("_BLOCK_ALPHA") )

            bloc_i = scene.visuals.Box( scal_i, scal_i, scal_i,  
                                        color = colr_i, edge_color="white" , )
            bloc_i.transform = transforms.STTransform( translate = xfrm_i[:3,3] )
            rtnGeo.append( bloc_i )
    if objReading.prob > 0.0:
        scl  = env_var("_BLOCK_SCALE") * objReading.prob
        bClr = env_var("_CLR_TABLE")[ objReading.label[:3] ]
        bClr.append( env_var("_BLOCK_ALPHA") )
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



########## RENDER MEMORY ###########################################################################

def render_memory_list( objs : list[GraspObj] ):
    """ Render the memory """
    objLst = reading_list_geo( objs )
    vispy_geo_list_window( objLst )



# def reading_dict_geo( objReading, lineColor = None, baseAlpha = 1.0 ):
#     """ Get geo for a single observation """
#     if lineColor is None:
#         lineColor = "black"
#     hasLabel  = ('label' in objReading)
#     labelSort = zip_dict_sorted_by_decreasing_value( objReading['labels'] )
#     objXfrm   = extract_pose_as_homog( objReading['pose'], noRot = True )
#     objPosn   = posn_from_xform( objXfrm ) #- hf
#     if hasLabel:
#         blcColor = env_var("_CLR_TABLE")[ objReading['label'][:3] ]
#     else:
#         blcColor = env_var("_CLR_TABLE")[ labelSort[0][0][:3]     ]

#     blcColor = (np.array( blcColor )*baseAlpha + np.array( [1,1,1,] )*(1.0-baseAlpha)).tolist()

#     block = scene.visuals.Box( env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), 
#                                color = blcColor, edge_color = lineColor )
#     block.transform = transforms.STTransform( translate = objPosn )

#     clnDct = dict()
#     for k, v in objReading['labels'].items():
#         clnDct[ k[:3] ] = np.round( v, 2 )

#     text = scene.visuals.Text(
#         str( clnDct ), 
#         # parent = block,
#         color  = 'black',
#     )
#     text.font_size = 3
#     # text.pos = [_BLOCK_SCALE*1.5, _BLOCK_SCALE*1.5, 0.0,]
#     text.pos = np.add( objPosn, [0.0, 0.0, env_var("_BLOCK_SCALE")*0.65,] )

#     return block, text
    
    

    