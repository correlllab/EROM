########## INIT ####################################################################################
import os

from pprint import pprint

import numpy as np

from magpie_control.ur5 import UR5_Interface
from aspire.env_config import env_var
from aspire.BlocksTask import set_blocks_env
from aspire.actions.pdls_behaviors import grasp_pose_from_posn

from env_config import dummy_object, KNOWN_BLOCKS
from draw_beliefs import render_memory_list
from dh_mp import UR5_DH
from LUMP import LUMP, plot_DH_robot



########## MAIN ####################################################################################
if __name__ == "__main__":
    set_blocks_env()
    rbt   = UR5_Interface()
    mp    = LUMP( [0.0 for _ in range(6)], rbt )
    objs  = KNOWN_BLOCKS()

    if 0:
        shots = mp.plan_3d_shots( objs, 0.250, 3, 60/80.0*np.pi )
        q     = mp.IK( shots[0] )
        urGeo = plot_DH_robot( UR5_DH, q )

        render_memory_list( syms = objs, robotPose = shots, xtra = urGeo )

        # pprint( shots )


    wp0    = grasp_pose_from_posn( [-0.5, 0.070, 0.5*env_var('_BLOCK_SCALE')] )
    wp1    = grasp_pose_from_posn( [ 0.5, 0.070, 0.5*env_var('_BLOCK_SCALE')] )
    wpLst  = mp.make_path_safe( wp0, wp1 )
    solns  = [mp.IK( wp ) for wp in wpLst]
    print( solns )
    rbtGeo = list()
    try:
        for soln in solns:
            rbtGeo.extend( plot_DH_robot( UR5_DH, soln ) )
    except AttributeError as e:
        print(e)

    render_memory_list( syms = objs, robotPose = wpLst, xtra = rbtGeo )


    os.system( 'kill %d' % os.getpid() ) 