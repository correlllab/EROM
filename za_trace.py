"""
########## QUESTIONS TO ANSWER #####################################################################
[?] Did the recording work?
[?] How stable are the symbols? Do the memories they are derived from always change?
[?] Do the readings that become symbols get erased after the action? 
[?] Do the symbol probabilities trend upward?
[?] Do the symbol quality scores trend upward?
[?] What is the impact of artifically boosting the quality of moved readings? Did it help?
"""


########## INIT && LOAD DATA #######################################################################
import pickle, os
from pprint import pprint

import numpy as np

from aspire.symbols import ObjPose, GraspObj, extract_pose_as_homog
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import set_experiment_env
from draw_beliefs import ( set_render_env, render_memory_list, render_scan_list, vispy_geo_list_window, 
                           table_geo, cpcd_geo )

# path = "/home/james/EROM/data/EROM-Memories_11-05-2024_13-43-39.pkl"
# path = "/home/james/EROM/data/EROM-Memories_11-06-2024_18-09-37.pkl"
path = "/home/will/james/erom/data/EROM-Memories_11-07-2024_16-10-32.pkl"
data = list()
with open( path, 'rb' ) as f:
    data = pickle.load( f )

set_blocks_env()
set_experiment_env()
set_render_env()


########## HELPER FUNCTIONS ########################################################################

def get_symbol_parents( objLst : list[GraspObj] ):
    """ Filter out the objects that are parents of symbols """
    rtnLst = list()
    for obj in objLst:
        if obj.SYM:
            rtnLst.append( obj )
    return rtnLst



########## ANALYSIS ################################################################################



""" ##### [Y] Did the recording work?: Yes ##################################### """
# Print all metadata
for i, datum in enumerate( data ):
    print( f"{datum['t']}:{i}: {datum['msg']}, {type(datum['data'])}" )



""" [?] How stable are the symbols? Do the memories they are derived from always change? """
""" [?] Do the symbol probabilities trend upward? """
""" [?] Do the symbol quality scores trend upward? """
""" [?] What is the impact of artifically boosting the quality of moved readings? Did it help? """
# FIXME: PLOTS FOR ABOVE QUESTIONS





########## DRAW SCANS ##############################################################################

totMem  = list()
camPose = np.eye(4)

for i, datum in enumerate( data ):

    # if datum['msg'] == 'meta':
    #     print( '.' )

    if datum['msg'] == 'camera':
        camPose = datum['data']
    if datum['msg'] == 'memory':
        readings = datum['data']['scan']
        for rdg in readings:
            rdg.cpcd.transform( camPose ) 
            rdg.pose = ObjPose( np.dot( camPose, extract_pose_as_homog( rdg.pose ) ) )
        totMem.extend( readings )
        # break

# rdng = totMem[-5]

# vispy_geo_list_window(
#     # table_geo(), 
#     cpcd_geo( rdng, camPose ),
# )

render_scan_list( totMem )


os.system( 'kill %d' % os.getpid() ) 