########## INIT ####################################################################################
import os

from pprint import pprint

import numpy as np

from magpie_control.ur5 import UR5_Interface
from aspire.BlocksTask import set_blocks_env

from env_config import dummy_object, KNOWN_BLOCKS
from draw_beliefs import render_memory_list
from dh_mp import UR5_DH
from LUMP import LUMP, plot_DH_robot



########## MAIN ####################################################################################
if __name__ == "__main__":
    set_blocks_env()
    rbt   = UR5_Interface()
    mp    = LUMP( [0.0 for _ in range(6)], rbt )
    # objs  = [dummy_object(),]
    objs  = KNOWN_BLOCKS()
    shots = mp.plan_3d_shots( objs, 0.250, 3, 60/80.0*np.pi )
    q     = mp.IK( shots[0] )
    urGeo = plot_DH_robot( UR5_DH, q )

    render_memory_list( syms = objs, robotPose = shots, xtra = urGeo )

    pprint( shots )

    os.system( 'kill %d' % os.getpid() ) 