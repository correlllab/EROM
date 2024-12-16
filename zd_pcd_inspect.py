########## INIT && LOAD DATA #######################################################################
import pickle, os
from pprint import pprint
from collections import deque

import numpy as np

from aspire.symbols import ObjPose, GraspObj, extract_pose_as_homog
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import set_experiment_env
from draw_beliefs import ( set_render_env, render_memory_list, scan_geo, vispy_geo_list_window, 
                           table_geo, cpcd_geo )

# path = "/home/james/EROM/data/EROM-Memories_11-05-2024_13-43-39.pkl"
# path = "/home/james/EROM/data/EROM-Memories_11-06-2024_18-09-37.pkl"
# path = "/home/will/james/erom/data/EROM-Memories_11-07-2024_16-10-32.pkl"
# path = "/home/will/james/erom/data/EROM-Memories_11-07-2024_16-24-41.pkl"
# path = "/home/will/james/erom/data/EROM-Memories_11-07-2024_16-29-50.pkl"
# path = "/home/will/james/erom/data/EROM-Memories_11-07-2024_16-36-58.pkl"
# path = "/home/will/james/erom/data/EROM-Memories_11-07-2024_16-42-59.pkl"
# path = "data/EROM-Memories_11-12-2024_16-54-59.pkl"
# path = "data/EROM-Memories_12-14-2024_19-47-38.pkl"
# path = "data/EROM-Memories_12-15-2024_20-05-38.pkl"
# path = "data/EROM-Memories_12-15-2024_20-14-39.pkl"
path = "data/EROM-Memories_12-15-2024_20-26-29.pkl"
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

_FETCH_CAM = False
_DRAW_SYMB = False
_DRAW_PCDS = True

for i, datum in enumerate( data ):

    # if datum['msg'] == 'meta':
    #     print( '.' )

    if ((_FETCH_CAM or _DRAW_PCDS) and (datum['msg'] == 'camera')):
        camPose = datum['data'].copy()
        # print( datum['data'].keys() )

    if _DRAW_PCDS and (datum['msg'] == 'memory'):
        readings = datum['data']['scan']
        totGeo   = deque()
        for rdg in readings:
            if len( rdg.cpcd ) > 20:
                totGeo.extend( cpcd_geo( rdg, camPose ) )
                totGeo.extend( scan_geo( rdg ) )

        vispy_geo_list_window( list( totGeo ) )

        # for rdg in readings:
        #     # rdg.cpcd.transform( camPose ) 
        #     # rdg.pose = ObjPose( np.dot( camPose, extract_pose_as_homog( rdg.pose ) ) )
        # # totMem.extend( readings )
        
    if _DRAW_SYMB and (datum['msg'] == 'symbols'):
        # render_memory_list( syms = datum['data']['scan'] )
        render_memory_list( syms = datum['data'] )

# rdng = totMem[-5]

# vispy_geo_list_window(
#     # table_geo(), 
#     cpcd_geo( rdng, camPose ),
# )

# render_scan_list( totMem )


os.system( 'kill %d' % os.getpid() ) 