########## INIT ####################################################################################
import os

from pprint import pprint

from magpie_control.ur5 import UR5_Interface
from aspire.BlocksTask import set_blocks_env

from env_config import dummy_object

from LUMP import LUMP



########## MAIN ####################################################################################
if __name__ == "__main__":
    set_blocks_env()
    rbt   = UR5_Interface()
    mp    = LUMP( [0.0 for _ in range(6)], rbt )
    objs  = [dummy_object(),]
    shots = mp.plan_3d_shots( objs, 0.250, 4 )

    pprint( shots )

    os.system( 'kill %d' % os.getpid() ) 