########## INIT ####################################################################################
import pythreejs as p3js
from IPython.display import display
import numpy as np

### ASPIRE ###
from aspire.env_config import env_var, env_sto
from aspire.homog_utils import homog_xform
from aspire.symbols import extract_pose_as_homog, GraspObj, CPCD

### Local ###
from utils import zip_dict_sorted_by_decreasing_value, get_pose_attr

_GOLDEN     = 1.618
_TABLE_THIC = 0.015

########## ENVIRONMENT #############################################################################

def set_render_env():
    """ Set vars used to draw EROM memories """
    env_sto( "_SCAN_ALPHA", 0.5 )



########## PY.THREE.JS #############################################################################

def get_p3js_matx( xform : np.ndarray ):
    transformation_matrix = p3js.Matrix4()
    # PYTHREEJS IS COLUMN-MAJOR
    transformation_matrix.set( xform.T.flatten().tolist() )
    return transformation_matrix


def rgb_to_html( rgb ):
    """ Convert an RGB color tuple to HTML hex color format. """
    # Extract RGB values (ignore alpha if present)
    r, g, b = rgb[:3]
    
    # Convert from [0,1] to [0,255] and then to hex
    r_int = int(round(r * 255))
    g_int = int(round(g * 255))
    b_int = int(round(b * 255))
    
    # Clamp values to [0, 255] range
    r_int = max(0, min(255, r_int))
    g_int = max(0, min(255, g_int))
    b_int = max(0, min(255, b_int))
    
    # Format as hex color
    return f'#{r_int:02x}{g_int:02x}{b_int:02x}'


def get_color_matl( color ):
    if not isinstance( color, str ):
        color = rgb_to_html( color )
    return p3js.MeshLambertMaterial( color = color )


def get_mesh_from_geo_color( geo, color ):
    return p3js.Mesh( geometry = geo, material = get_color_matl( color ) )

def np_arr_as_tuple( arr : np.ndarray ):
    return tuple( arr.tolist() )

########## DISPLAY WINDOW ##########################################################################

def p3js_geo_list_window( geoLst : list, robotPose = None, xtra = None ):

    def add_pose( pose ):
        nonlocal geoLst
        rbtAxs = p3js.AxesHelper( size = 0.15 )
        rbtAxs.matrixAutoUpdate = False
        rbtAxs.matrix = get_p3js_matx( pose )
        geoLst.append( rbtAxs )

    if isinstance( robotPose, np.ndarray ):
        add_pose( robotPose )
    elif isinstance( robotPose, list ):
        for pose in robotPose:
            add_pose( pose )

    axes = p3js.AxesHelper( size = 0.15 )

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
    # camera.lookAt([
    #     0.0, 
    #     env_var("_MIN_Y_OFFSET") + env_var("_Y_WRK_SPAN")/2.0, 
    #     0.0,
    # ])

    # Add lighting (optional, but makes it look better)
    key_light = p3js.DirectionalLight(color='white', position=[1, 1, 1], intensity=0.5)
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
        controls =[p3js.OrbitControls(controlling=camera)],
        width    = int(600*_GOLDEN),
        height   = int(600)
    )

    # Display
    display( renderer )

########## DRAWING FUNCTIONS #######################################################################

def solid_box_geo( xScl, yScl, zScl, color = None ):
    """ Draw a wireframe cuboid """
    if color is None:
        color = [0,1,0,1]
    # Create cube geometry
    boxGeo  = p3js.BoxGeometry( xScl, yScl, zScl )
    print( type( boxGeo ) )
    boxMesh = get_mesh_from_geo_color( boxGeo, color = color )
    print( type( boxMesh ) )
    return boxMesh


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
        color = rgb_to_html( color )
    
    # Create cube geometry
    geometry = p3js.BoxGeometry( xScl, yScl, zScl )
    
    # Create wireframe material
    material = p3js.MeshBasicMaterial(
        color=color,
        wireframe=True,
        wireframeLinewidth=3
    )
    
    # Create mesh
    cube = p3js.Mesh(geometry=geometry, material=material)

    return cube


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


def symbol_geo( sym : GraspObj ):
    objXfrm = extract_pose_as_homog( sym, noRot = True )
    wf1 = wireframe_box_geo( env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), 
                             color = "black" )
    wf1.position = np_arr_as_tuple( objXfrm[:3,3] )
    wf2 = wireframe_box_geo( env_var("_BLOCK_SCALE")*1.125, env_var("_BLOCK_SCALE")*1.125, env_var("_BLOCK_SCALE")*1.125, 
                             color = "black" )
    wf2.position = np_arr_as_tuple( objXfrm[:3,3] )
    scl  = env_var("_BLOCK_SCALE") * 0.200
    bClr = env_var("_CLR_TABLE")[ sym.label[:3] ]
    bClr.append( env_var("_BLOCK_ALPHA") )
    blc  = solid_box_geo( scl, scl, scl, color = bClr )
    blc.position = np_arr_as_tuple( objXfrm[:3,3] )
    return [wf1, wf2, blc,] 


def target_geo( sym : GraspObj, colorName : str = 'black' ):
    objXfrm = get_pose_attr( sym )
    wf1 = wireframe_box_geo( env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), env_var("_BLOCK_SCALE"), color = colorName )
    wf1.position = np_arr_as_tuple( objXfrm[:3,3] )
    rtnLst = [wf1,]
    return rtnLst



########## RENDER MEMORY ###########################################################################

def reading_list_geo( objs : list[GraspObj] ):
    """ Get geo for a list of observations """
    rtnGeo = [table_geo(),]
    for obj in objs:
        rtnGeo.extend( reading_geo( obj ) )
    return rtnGeo


def symbol_list_geo( objs : list[GraspObj], noTable = True ):
    """ Get geo for a list of symbols """
    if noTable:
        rtnGeo = list()
    else:
        rtnGeo = [table_geo(),]
    for obj in objs:
        rtnGeo.extend( symbol_geo( obj ) )
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
    
    # Add lighting (optional, but makes it look better)
    key_light = p3js.DirectionalLight(color='white', position=[3, 5, 1], intensity=0.5)
    ambient_light = p3js.AmbientLight(color='#404040')
    
    # Create scene
    scene = p3js.Scene(
        children=[cube, camera, key_light, ambient_light],
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