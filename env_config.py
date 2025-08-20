########## INIT ####################################################################################
import numpy as np
from aspire.env_config import env_var, env_sto
from aspire.symbols import ObjPose, GraspObj
from Memory import now



########## SETUP ENV ###############################################################################

def BASE_TARGET():
    _poseGrn = np.eye(4)
    _poseGrn[0:3,3] = [ -0.200, # env_var("_MIN_X_OFFSET")+env_var("_X_WRK_SPAN")/2.0, 
                        -0.300, # env_var("_MIN_Y_OFFSET")+env_var("_Y_WRK_SPAN")/2.0, 
                         0.5*env_var("_BLOCK_SCALE")+env_var("_Z_TABLE"), ]
    return ObjPose( _poseGrn )



def KNOWN_BLOCKS():
    """ Set block positions """
    yOfst = 0.025
    bloc1 = np.array([[-0.994,  0.111,  0.022, -0.38 ],
                    #   [ 0.112,  0.993,  0.044,  0.003+yOfst+yOfst],
                      [ 0.112,  0.993,  0.044,  0.003+yOfst],
                      [-0.017,  0.046, -0.999,  0.246],
                      [ 0.0  ,  0.0  ,  0.0  ,  1.0  ],])
    bloc2 = np.array([[-0.994,  0.111,  0.022, -0.38 ],
                      [ 0.112,  0.993,  0.044, -0.102+yOfst],
                      [-0.017,  0.046, -0.999,  0.246],
                      [ 0.0  ,  0.0  ,  0.0  ,  1.0  ],])
    bloc3 = np.array([[-0.994,  0.111,  0.022, -0.38 ],
                      [ 0.112,  0.993,  0.044, -0.199+yOfst],
                      [-0.017,  0.046, -0.999,  0.246],
                      [ 0.0  ,  0.0  ,  0.0  ,  1.0  ],])
    names = ['redBlock','grnBlock','bluBlock',]
    rtnBlocks = list()
    for i, bloc_i in enumerate([bloc1,bloc2,bloc3,]):
        pose_i = np.eye(4)
        pose_i[0:3,3] = bloc_i[0:3,3]
        pose_i[2,3]   = env_var("_BLOCK_SCALE") / 2.0
        obj_i         = GraspObj(
            label = names[i],
            pose  = pose_i
        )
        rtnBlocks.append( obj_i )
        env_sto( f"_KNOWN_BLOCK_{i}", obj_i ) 
    return rtnBlocks

def dummy_object():
    """ Use as target for first-pass sensory planning """
    return GraspObj(
        labels = {'grnBlock':1.0,}, 
        pose   = BASE_TARGET(), 
        ts     = now(), 
        count  = 1, 
        score  = 0.0,
    )


def set_experiment_env():
    """ Params for this experiment """

    env_sto( "_SPACE_EXPAND",  0.000 ) 

    env_sto( "_MIN_X_OFFSET", -0.468 - env_var( "_SPACE_EXPAND" ) )
    env_sto( "_MAX_X_OFFSET", -0.103 + env_var( "_SPACE_EXPAND" ) )
    env_sto( "_MIN_Y_OFFSET", -0.625 - env_var( "_SPACE_EXPAND" ) ) 
    env_sto( "_MAX_Y_OFFSET",  0.200 + env_var( "_SPACE_EXPAND" ) )

    _THRESH_FACTOR = 0.0625

    env_sto( "_OWL2_THRESH"  ,    0.005*_THRESH_FACTOR ) # 0.005
    env_sto( "_SEG_MAX_HITS"    , 2     ) # 1 # 3 # 5 # 10 # 50
    env_sto( "_SEG_SCORE_THRESH", 0.100*_THRESH_FACTOR ) # 0.025 # 0.075 # 0.100
    env_sto( "_OWL2_TOPK"       , 1     )

    env_sto( "_USE_SAM2_SAMPLES", False ) 
    # env_sto( "_USE_SAM2_SAMPLES", True ) # 2025-07-31: ?? BAD ??

    env_sto( "_Z_SAFE", 0.350 )

     # 3D Printed Blocks

    KNOWN_BLOCKS()
    _trgtGrn = BASE_TARGET()
    
    env_sto( "_BLOCK_VOLUME", env_var( "_BLOCK_SCALE" )**3 )

    env_sto( "_VERBOSE"     , True )
    env_sto( "_USE_GRAPHICS", False )
    env_sto( "_SCAN_ALPHA"  , 0.35  )
    # env_sto( "_SHOW_SEGMENT", True )
    env_sto( "_SHOW_SEGMENT", False )

    env_sto( "_USE_Z_SNAP"   , False )
    env_sto( "_Z_SNAP_CHEAT" , True  )

    # env_sto( "_Z_SNAP_BOOST" , -0.25*env_var("_BLOCK_SCALE")   )
    # env_sto( "_Z_SNAP_BOOST" , 0.00*env_var("_BLOCK_SCALE") )
    env_sto( "_Z_SNAP_BOOST" , 0.125*env_var("_BLOCK_SCALE") )
    # env_sto( "_Z_SNAP_BOOST" , 0.25*env_var("_BLOCK_SCALE") )

    env_sto( "_Z_STACK_BOOST", 0.00*env_var("_BLOCK_SCALE") )
    # env_sto( "_Z_STACK_BOOST", 0.125*env_var("_BLOCK_SCALE") )

    env_sto( "_N_INTAKE_SCANS", 1 )

    env_sto( "_N_XTRA_SPOTS",   3     )
    env_sto( "_N_REQD_OBJS" ,   3     )
    env_sto( "_CONFUSE_PROB",   0.025 )

    # env_sto( "_BAYES_RAD_L2_M" , 1.000*env_var("_BLOCK_SCALE")  )
    # env_sto( "_BAYES_RAD_L2_M" , 0.950*env_var("_BLOCK_SCALE")  )
    # env_sto( "_BAYES_RAD_L2_M" , 0.900*env_var("_BLOCK_SCALE")  )
    env_sto( "_BAYES_RAD_L2_M" , 0.800*env_var("_BLOCK_SCALE")  ) # 2025-03-11: ?? WINNING PARAMS ??
    # env_sto( "_BAYES_RAD_L2_M" , 0.750*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.700*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.650*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.500*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.400*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.350*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.300*env_var("_BLOCK_SCALE")  ) 
    # env_sto( "_BAYES_RAD_L2_M" , 0.250*env_var("_BLOCK_SCALE")  ) # 2025-02-24: ?? WINNING PARAMS ??

    env_sto( "_PLACE_XY_ACCEPT", 0.400*env_var("_BLOCK_SCALE")  )
    # env_sto( "_PLACE_XY_ACCEPT", 0.600*env_var("_BLOCK_SCALE")  )

    # env_sto( "_WIDE_XY_ACCEPT" , 0.750*env_var("_BLOCK_SCALE")  )
    env_sto( "_WIDE_XY_ACCEPT" , 0.900*env_var("_BLOCK_SCALE")  )


    env_sto( "_WIDE_COLLIDE"   , env_var("_BAYES_RAD_L2_M")  ) # 2025-02-25: ?? WINNING PARAMS ??

    # env_sto( "_WIDE_COLLIDE"   , 0.450*env_var("_BLOCK_SCALE")  )
    # env_sto( "_WIDE_COLLIDE"   , 0.625*env_var("_BLOCK_SCALE")  )
    # env_sto( "_WIDE_COLLIDE"   , 0.800*env_var("_BLOCK_SCALE")  )
    # env_sto( "_WIDE_COLLIDE"   , 0.950*env_var("_BLOCK_SCALE")  )
    # env_sto( "_WIDE_COLLIDE"   , 1.125*env_var("_BLOCK_SCALE")  ) # WHY WOULD I EVEN DO THAT?

    env_sto( "_WIDE_PLACEMENT" , 3.000*env_var("_BLOCK_SCALE") ) # 2025-02-25: ?? WINNING PARAMS ??


    env_sto( "_WIDE_Z_ABOVE"         , 1.75*env_var("_BLOCK_SCALE") )
    env_sto( "_SMUSH_Z_ABOVE"        , 0.80*env_var("_BLOCK_SCALE") )
    env_sto( "_TABLE_SUPPORT_Z_MAX"  , 0.65*env_var("_BLOCK_SCALE") )
    env_sto( "_DEFAULT_TABLE_SUPPORT", True )

    env_sto( "_ROBOT_FREE_SPEED", 0.125 * 2.00 ) # 2.00 # 2.50 # 3.00 # 3.50
    env_sto( "_ROBOT_HOLD_SPEED", 0.125 * 1.50 ) # 1.50 # 2.00
    env_sto( "_ROBOT_LIN_ACCEL" , 0.500 * 1.00 ) # 0.75 # 1.00 # 1.25 # 1.50

    env_sto( "_ACCEPT_POSN_ERR" , 1.00*env_var( "_BLOCK_SCALE" ) ) # 0.60 # 0.75 # 0.90
    
    env_sto( "_GOAL_GRB" ,
        ( 'and',
            ('GraspObj', 'grnBlock' , _trgtGrn ), # ; Tower
            ('Supported', 'redBlock', 'grnBlock'), 
            ('Supported', 'bluBlock', 'redBlock'), 
            ('HandEmpty',),
        )        
    )


    env_sto( "_GOAL_OR_RGB" ,
        ( 'or',
            ## Red Base ##
            ( 'and',
                ('GraspObj', 'redBlock', _trgtGrn  ), # ; Tower
                ('Supported', 'grnBlock', 'redBlock', ), 
                ('Supported', 'bluBlock', 'grnBlock', ), 
                ('HandEmpty',),
            ),   
            ( 'and',
                ('GraspObj', 'redBlock', _trgtGrn  ), # ; Tower
                ('Supported', 'bluBlock', 'redBlock', ), 
                ('Supported', 'grnBlock', 'bluBlock', ), 
                ('HandEmpty',),
            ),   
            ## Green Base ##
            ( 'and',
                ('GraspObj', 'grnBlock' , _trgtGrn  ), # ; Tower
                ('Supported', 'redBlock', 'grnBlock', ), 
                ('Supported', 'bluBlock', 'redBlock', ), 
                ('HandEmpty',),
            ),   
            ( 'and',
                ('GraspObj', 'grnBlock' , _trgtGrn  ), # ; Tower
                ('Supported', 'bluBlock', 'grnBlock', ), 
                ('Supported', 'redBlock', 'bluBlock', ), 
                ('HandEmpty',),
            ),   
            ## Blue Base ##
            ( 'and',
                ('GraspObj', 'bluBlock',  _trgtGrn  ), # ; Tower
                ('Supported', 'redBlock', 'bluBlock', ), 
                ('Supported', 'grnBlock', 'redBlock', ), 
                ('HandEmpty',),
            ),   
            ( 'and',
                ('GraspObj', 'bluBlock', _trgtGrn  ), # ; Tower
                ('Supported', 'grnBlock', 'bluBlock', ), 
                ('Supported', 'redBlock', 'grnBlock', ), 
                ('HandEmpty',),
            ),   
        )
            
    )


    env_sto( "_GOAL_RRR" ,
        ( 'and',
            ('GraspObj',  1 , _trgtGrn  ), # ; Tower
            ('Supported', 2, 1), 
            ('Supported', 3, 2), 
            ('HandEmpty',),
            ('ObjLabel', 1, 'redBlock'),
            ('ObjLabel', 2, 'redBlock'),
            ('ObjLabel', 3, 'redBlock'),
        )
    )

    env_sto( "_GRIPPER_TORQUE", 250 ) 

    env_sto( "_UPDATE_PERIOD_S", 3.0       ) 
    env_sto( "_OBJ_TIMEOUT_S"  , 60.0*10.0 )

    env_sto( "_SCORE_FILTER_EXP", 0.85 )

    # env_sto( "_UPDATE_FRAC", 0.25 )
    env_sto( "_UPDATE_FRAC", 0.35 )
    # env_sto( "_UPDATE_FRAC", 0.45 )
    # env_sto( "_UPDATE_FRAC", 0.85 )

    # env_sto( "_NULL_EVIDENCE" , True )
    env_sto( "_NULL_EVIDENCE" , False ) # 2025-02-25: ?? WINNING PARAMS ??


    env_sto( "_REPAIR_BAYES" , True ) 


    env_sto( "_DEF_NULL_SCORE", 1.00 )

    # env_sto( "_NULL_THRESH"   , 0.50 )
    # env_sto( "_NULL_THRESH"   , 0.60 )
    # env_sto( "_NULL_THRESH"   , 0.65 )
    env_sto( "_NULL_THRESH"   , 0.75 ) # 2025-02-24: ?? WINNING PARAMS ??
    # env_sto( "_NULL_THRESH"   , 0.95 )

    env_sto( "_GRASP_NUDGE_M", -0.005 )

    env_sto( "_USE_SPACE_HACK", False )
    env_sto( "_USE_PERC_HACK" , True )
    

    env_sto( "_USE_POSE_CHEAT", True )
    env_sto( "_CHEAT_LABEL"   , False and env_var("_USE_POSE_CHEAT") )
    env_sto( "_CHEAT_POSE"    , False and env_var("_USE_POSE_CHEAT") )
    env_sto( "_CHEAT_LKG"     , False and env_var("_USE_POSE_CHEAT") )


    env_sto( "_N_PERC_SHOTS"    , 3 )
    env_sto( "_N_PATH_DIV"      , 2 ) # 4 # 6
    env_sto( "_PERC_MULT_FACTOR", 2 ) # 3 # 8
    env_sto( "_N_SEARCH_SHOTS"  , 3 )

    env_sto( "_N_GRID_EXPAND"   , 3 ) 
    env_sto( "_SEARCH_GRID_UNIT", 0.030 ) # 0.050

    env_sto( "_N_GRID_HALF_PTS" , 2 )
    env_sto( "_RETRY_CLOUDS"    , True )