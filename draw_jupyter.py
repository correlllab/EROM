########## INIT ####################################################################################
from collections import deque

import pythreejs as p3js
from IPython.display import display
import numpy as np

### ASPIRE ###
from aspire.env_config import env_var, env_sto
from aspire.homog_utils import homog_xform
from aspire.symbols import extract_pose_as_homog, GraspObj, CPCD

### Local ###
from utils import zip_dict_sorted_by_decreasing_value, get_pose_attr, parse_action
from homog_utils import posn_from_xform
from Reader import EROM_Reader

_GOLDEN     = 1.618
_TABLE_THIC = 0.015
_AXES_SCALE = 0.150
_WIRE_THICK = 3
_SPHERE_DIV_LONG = 16
_SPHERE_DIV_LAT  =  8


########## ENVIRONMENT #############################################################################

def set_render_env():
    """ Set vars used to draw EROM memories """
    env_sto( "_SCAN_ALPHA", 0.5 )



########## PY.THREE.JS #############################################################################

def get_p3js_matx( xform : np.ndarray ):
    """ Convert Numpy row-major matrix to PyThreeJS column-major matrix """
    # transformation_matrix = p3js.Matrix4()
    # PYTHREEJS IS COLUMN-MAJOR
    # transformation_matrix.set( value = xform.T.flatten().tolist() )
    # transformation_matrix.value = xform.T.flatten().tolist()
    return xform.T.flatten().tolist()
    # return transformation_matrix


def clamp( n, lower, upper ):
    """ Return `n` bounded by [`lower`, `upper`] """
    return min( max( n, lower ), upper )


def rgb_to_html_alpha( rgb ):
    """ Convert an RGB color tuple to HTML hex color format. """
    # Extract RGB values (ignore alpha if present)
    r, g, b = rgb[:3]
    if len( rgb ) > 3:
        alpha = clamp( rgb[3], 0.0, 1.0 )
    else:
        alpha = 1.0
    # Convert from [0,1] to [0,255] and then to hex
    r_int = int(round(r * 255))
    g_int = int(round(g * 255))
    b_int = int(round(b * 255))
    # Clamp values to [0, 255] range
    r_int = max(0, min(255, r_int))
    g_int = max(0, min(255, g_int))
    b_int = max(0, min(255, b_int))
    # Format as hex color
    return f'#{r_int:02x}{g_int:02x}{b_int:02x}', alpha


def get_color_matl( color, matlType = "lambert", wireframe = False ):
    """ Get a material associated with this color as a one of several PyThreeJS Materials """
    alpha = 1.0
    if not isinstance( color, str ):
        color, alpha = rgb_to_html_alpha( color )
    if matlType == "lambert":
        return p3js.MeshLambertMaterial( 
            color       = color, 
            transparent = (alpha < 1.0),
            opacity     = alpha,
            wireframe   = wireframe,
            wireframeLinewidth = _WIRE_THICK
        )
    elif matlType == "basic":
        return p3js.MeshBasicMaterial( 
            color       = color, 
            transparent = (alpha < 1.0),
            opacity     = alpha,
            wireframe   = wireframe,
            wireframeLinewidth = _WIRE_THICK
        )
    else:
        raise ValueError( f"NO MATERIAL TYPE: {matlType}" )


def get_line_matl( color, weight = 5 ):
    """ Get a material for drawing line segments """
    alpha = 1.0
    if not isinstance( color, str ):
        color, alpha = rgb_to_html_alpha( color )
    return p3js.LineMaterial( 
        linewidth   = weight, 
        color       = color,
        transparent = (alpha < 1.0),
        opacity     = alpha
    )


def get_mesh_from_geo_color( geo, color ):
    """ Assign geometry and a material to a PyThreeJS mesh for drawing """
    return p3js.Mesh( geometry = geo, material = get_color_matl( color ) )


def np_arr_as_tuple( arr : np.ndarray ):
    """ Format Numpy coordinates in the way that PyThreeJS likes, apparently """
    return tuple( arr.tolist() )



########## DISPLAY WINDOW ##########################################################################

def p3js_geo_list_window( geoLst : list, robotPose = None, xtra = None ):
    """ Display the given geometry list in a Jupyter widget viewport """

    def add_pose( pose ):
        """ Display right-hand cartesian axes at the given row-major homogeneous pose """
        nonlocal geoLst
        rbtAxs = p3js.AxesHelper( size = _AXES_SCALE/2.0 )
        rbtAxs.matrixAutoUpdate = False
        rbtAxs.matrix = get_p3js_matx( pose )
        geoLst.append( rbtAxs )

    # Display effector poses associated with the scene
    if isinstance( robotPose, np.ndarray ):
        add_pose( robotPose )
    elif isinstance( robotPose, list ):
        for pose in robotPose:
            add_pose( pose )

    # Display the lab frame origin
    axes = p3js.AxesHelper( size = _AXES_SCALE )

    # Anything else that might be fun to draw
    if isinstance( xtra, list ):
        geoLst.extend( xtra )

    # Set up camera
    camera = p3js.PerspectiveCamera(
        position= tuple([
            env_var("_MIN_X_OFFSET"), 
            env_var("_MIN_Y_OFFSET")*0.75, 
            env_var("_X_WRK_SPAN")/2.0,
        ]),
        up     = [0, 0, 1,],
        fov    = 70,
        aspect = _GOLDEN,
        near   = 0.1,
        far    = 10
    )
    # camera.lookAt([ # DOES NOT WORK???
    #     0.0, 
    #     env_var("_MIN_Y_OFFSET") + env_var("_Y_WRK_SPAN")/2.0, 
    #     0.0,
    # ])

    # Add lighting (optional, but makes it look better)
    key_light     = p3js.DirectionalLight(color='white', position=[1, 1, 1], intensity=0.5)
    ambient_light = p3js.AmbientLight(color='#40404f')

    geoLst.extend( [axes, camera, key_light, ambient_light] )

    # Create scene
    scene = p3js.Scene(
        children   = geoLst,
        background = '#ffffff'
    )
    
    # Create renderer
    renderer = p3js.Renderer(
        camera   = camera,
        scene    = scene,
        controls = [p3js.OrbitControls(controlling=camera)],
        width    = int(600*_GOLDEN),
        height   = int(600)
    )

    # Display @ Jupyter
    display( renderer )



########## DRAWING FUNCTIONS #######################################################################

def solid_box_geo( xScl, yScl, zScl, color = None ):
    """ Draw a solid cuboid """
    if color is None:
        color = [0,1,0,1]
    # Create cube geometry
    boxGeo  = p3js.BoxGeometry( xScl, yScl, zScl )
    # print( type( boxGeo ) )
    boxMesh = get_mesh_from_geo_color( boxGeo, color = color )
    # print( type( boxMesh ) )
    return boxMesh


def solid_ball_geo( radius, color = None ):
    """ Draw a solid sphere """
    if color is None:
        color = [0,1,0,1]
    ballGeo  = p3js.SphereGeometry( radius, _SPHERE_DIV_LONG, _SPHERE_DIV_LAT )
    ballMesh = get_mesh_from_geo_color( ballGeo, color = color )
    return ballMesh


def table_geo():
    """ Draw the usable workspace """
    tableMesh = solid_box_geo( env_var('_X_WRK_SPAN'), env_var("_Y_WRK_SPAN"), _TABLE_THIC, color = [237/255.0, 139/255.0, 47/255.0, 1.0] )
    tableMesh.position =  (
        env_var("_MIN_X_OFFSET") + env_var("_X_WRK_SPAN")/2.0, 
        env_var("_MIN_Y_OFFSET") + env_var("_Y_WRK_SPAN")/2.0, 
        -_TABLE_THIC/2.0+env_var("_Z_TABLE")
    )
    return tableMesh


def wireframe_box_geo( xScl, yScl, zScl, color = None ):
    """ Draw a wireframe cuboid """
    if color is None:
        color = [0,0,0,1]
    if not isinstance( color, str ):
        color, alpha = rgb_to_html_alpha( color )
    # Create cube geometry
    geometry = p3js.BoxGeometry( xScl, yScl, zScl )
    # Create wireframe material
    material = get_color_matl( color, "basic", wireframe = True )
    # Create && Return mesh
    return p3js.Mesh( geometry = geometry, material = material )


def reading_geo( objReading : GraspObj, alpha = None ):
    """ Get geo for a single observation """
    if alpha is None:
        alpha = 1.0
    belClr = [0.5, 0.0, 1.0, alpha,]
    lkgClr = [1.0, 0.0, 0.0, alpha,]
    labelSort = zip_dict_sorted_by_decreasing_value( objReading.labels )
    objXfrm   = extract_pose_as_homog( objReading, noRot = True )
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
    wir.position = np_arr_as_tuple( objXfrm[:3,3] )
    rtnGeo.extend( [wir,] )
    
    for i in range(3):
        objXfrm[i,3] -= hf
    for i in range( 0, min( len(labelSort), len(topCrnrs) ) ):
        prob_i = labelSort[i][1]
        if (prob_i > 0.0):
            scal_i  = env_var("_BLOCK_SCALE") * prob_i
            xfrm_i = topCrnrs[i-1]
            xfrm_i = objXfrm.dot( xfrm_i )
            colr_i = env_var("_CLR_TABLE")[ labelSort[i][0][:3] ]
            colr_i.append( 1.0 )
            colr_i[-1] = env_var("_BLOCK_ALPHA") if (alpha is None) else alpha
            bloc_i = solid_box_geo( scal_i, scal_i, scal_i, color = colr_i )
            bloc_i.position = np_arr_as_tuple( xfrm_i[:3,3] ) 
            rtnGeo.append( bloc_i )
    if (objReading.prob > 0.0) and (objReading.label != env_var("_NULL_NAME")):
        scl  = env_var("_BLOCK_SCALE") * objReading.prob
        bClr = env_var("_CLR_TABLE")[ objReading.label[:3] ]
        bClr.append( 1.0 )
        bClr[-1] = env_var("_BLOCK_ALPHA") if (alpha is None) else alpha
        blc  = solid_box_geo( scl, scl, scl, color = bClr )
        blc.position = np_arr_as_tuple( objXfrm[:3,3] )
        rtnGeo.extend( [blc,] )
    return rtnGeo


def symbol_geo( sym : GraspObj, alpha = None ):
    """ Get geo for a determinized symbol, Scale of solid block expresses confidence in class """
    if alpha is None:
        alpha = env_var("_BLOCK_ALPHA")
    objXfrm = extract_pose_as_homog( sym, noRot = True )
    wf1 = wireframe_box_geo( env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), 
                             color = "black" )
    wf1.position = np_arr_as_tuple( objXfrm[:3,3] )
    wf2 = wireframe_box_geo( env_var("_BLOCK_SCALE")*1.125, env_var("_BLOCK_SCALE")*1.125, env_var("_BLOCK_SCALE")*1.125, 
                             color = "black" )
    wf2.position = np_arr_as_tuple( objXfrm[:3,3] )
    scl  = env_var("_BLOCK_SCALE") * 0.200
    bClr = env_var("_CLR_TABLE")[ sym.label[:3] ]
    bClr.append( alpha )
    blc  = solid_box_geo( scl, scl, scl, color = bClr )
    blc.position = np_arr_as_tuple( objXfrm[:3,3] )
    return [wf1, wf2, blc,] 


def target_geo( sym : GraspObj, colorName : str = 'black' ):
    """ Get geo for a presumptive block location """
    objXfrm = get_pose_attr( sym )
    wf1 = wireframe_box_geo( env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), color = colorName )
    wf1.position = np_arr_as_tuple( objXfrm[:3,3] )
    rtnLst = [wf1,]
    return rtnLst


def line_segments_geo( segCoordsList : list[list[list[float]]] | np.ndarray, color, weight ):
    """ Get geo for line segments from a list of endpoint pairs """
    if isinstance( segCoordsList, np.ndarray ):
        segCoordsList = segCoordsList.tolist()
    g2 = p3js.LineSegmentsGeometry( positions = segCoordsList )
    m2 = get_line_matl( color, weight )
    return p3js.LineSegments2( g2, m2 )



########## RENDER PLANS ############################################################################

def plan_step_geo( plan : dict, Zsafe : float = 0.250 ):
    """ Parse the plan and display it """
    # plan   = parse_action( planText )
    bgnPsn = posn_from_xform( plan["bgnPose"] )
    endPsn = posn_from_xform( plan["endPose"] )
    bgnUpP = bgnPsn.copy()
    bgnUpP[2] = Zsafe
    endUpP = endPsn.copy()
    endUpP[2] = Zsafe

    geo = line_segments_geo( 
        [ [bgnPsn, bgnUpP,],
          [bgnUpP, endUpP,],
          [endUpP, endPsn,],], 
        color  = env_var("_CLR_TABLE")[ plan["name"][:3] ], 
        weight = 10 
    )

    return [geo,]



########## RENDER MEMORY ###########################################################################

def reading_list_geo( objs : list[GraspObj], alpha = None ):
    """ Get geo for a list of observations """
    rtnGeo = [table_geo(),]
    for obj in objs:
        rtnGeo.extend( reading_geo( obj, alpha ) )
    return rtnGeo


def symbol_list_geo( objs : list[GraspObj], noTable = True, alpha = None ):
    """ Get geo for a list of symbols """
    if noTable:
        rtnGeo = list()
    else:
        rtnGeo = [table_geo(),]
    for obj in objs:
        rtnGeo.extend( symbol_geo( obj, alpha ) )
    return rtnGeo


def target_list_geo( objs : list[GraspObj], noTable = True ):
    """ Get geo for a list of symbols """
    if noTable:
        rtnGeo = list()
    else:
        rtnGeo = [table_geo(),]
    for obj in objs:
        rtnGeo.extend( target_geo( obj ) )
    return rtnGeo


def render_confusion_dict( inDct : dict ):
    """ Render the confusion between two timesteps """
    _MOVE_COLR  = "cyan"
    _MOVE_THIC  = 7
    _CURR_ALPHA = 0.750    
    _PREV_ALPHA = 0.375
    totGeo      = list()
    if ('moves' in inDct) and len( inDct['moves'] ):
        segmnt  = [[ posn_from_xform( inDct['moves'][0]['src'] ).tolist(), posn_from_xform( inDct['moves'][0]['dst'] ).tolist(), ],]
        segMesh = line_segments_geo( segmnt, _MOVE_COLR, _MOVE_THIC )
        totGeo.append( segMesh )
    if ('conflicts' in inDct) and len( inDct['conflicts'] ):
        for conflict in inDct['conflicts']:
            ballMesh = solid_ball_geo( env_var("_BLOCK_SCALE")*1.125/2.0, color = [1.0, 0.0, 0.0, 0.5,] )
            ballMesh.position = posn_from_xform( conflict )
            totGeo.append( ballMesh )
    if ('currObjs' in inDct) and len( inDct['currObjs'] ):
        totGeo.extend( symbol_list_geo( inDct['currObjs'], noTable = True, alpha = _CURR_ALPHA ) )
    if ('prevObjs' in inDct) and len( inDct['prevObjs'] ):
        totGeo.extend( symbol_list_geo( inDct['prevObjs'], noTable = False, alpha = _PREV_ALPHA ) )
    return p3js_geo_list_window( totGeo )


def render_memory_list( objs : list[GraspObj] = None, syms = None, removed = None, robotPose = None, xtra = None ):
    """ Render the memory """
    if objs is not None:
        objLst       = reading_list_geo( objs )
        missingTable = False
    else:
        objLst       = list()
        missingTable = True
    if syms is not None:
        objLst.extend( symbol_list_geo( syms, noTable = (not missingTable) ) )
    if removed is not None:
        objLst.extend( target_list_geo( removed ) )
    return p3js_geo_list_window( objLst, robotPose, xtra )


def render_state_and_plan_step( syms : list[GraspObj] = None, planDct : str = None ):
    """ Render the symbols and the next step """
    objLst = deque()
    if syms is not None:
        objLst.extend( symbol_list_geo( syms, noTable = False ) )
    if planDct is not None:
        objLst.extend( plan_step_geo( planDct ) )
    return p3js_geo_list_window( list( objLst ) )



########## TEST ####################################################################################

def display_wireframe_cube(size=1.0, color='#3380ff', camera_position=(2, 2, 2)):
    """
    Display a blue wireframe cube using pythreejs in Jupyter Lab.
    
    Parameters:
    -----------
    size : float
        Size of the cube edges
    color : str
        Color of the wireframe (hex color)
    camera_position : tuple
        Camera position (x, y, z)
    """
    # Create cube geometry
    geometry = p3js.BoxGeometry(width=size, height=size, depth=size)
    
    # Create wireframe material
    material = p3js.MeshBasicMaterial(
        color=color,
        wireframe=True,
        wireframeLinewidth=2
    )
    
    # Create mesh
    cube = p3js.Mesh(geometry=geometry, material=material)
    
    # Set up camera
    camera = p3js.PerspectiveCamera(
        position=camera_position,
        fov=50,
        aspect=1.0,
        near=0.1,
        far=1000
    )

    g2 = p3js.LineSegmentsGeometry(
        positions=[
            [[0, 0, 0], [1, 1, 1]],
            [[2, 2, 2], [4, 4, 4]]
        ],
    )
    m2 = p3js.LineMaterial(linewidth=10, color='cyan')
    line2 = p3js.LineSegments2(g2, m2)
    
    
    # Add lighting (optional, but makes it look better)
    key_light = p3js.DirectionalLight(color='white', position=[3, 5, 1], intensity=0.5)
    ambient_light = p3js.AmbientLight(color='#404040')
    
    # Create scene
    scene = p3js.Scene(
        children=[cube, line2, camera, key_light, ambient_light],
        background='#ffffff'
    )
    
    # Create renderer
    renderer = p3js.Renderer(
        camera=camera,
        scene=scene,
        controls=[p3js.OrbitControls(controlling=camera)],
        width=600,
        height=600
    )
    
    # Display
    display(renderer)
    
    return renderer, cube