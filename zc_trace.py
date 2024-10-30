"""
########## QUESTIONS TO ANSWER #####################################################################
[?] Did the recording work?
[?] How stable are the symbols? Do the memories they are derived from always change?
[?] Do the readings that become symbols get erased after the action? Why?
[?] Do the symbol probabilities trend upward?
[?] Do the symbol quality scores trend upward?
[?] What is the impact of artifically boosting the quality of moved readings? Did it help?
"""

########## INIT && LOAD DATA #######################################################################
import pickle

from aspire.symbols import ObjPose, GraspObj, extract_pose_as_homog
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import set_experiment_env

path = ""
data = list()
with open( path, 'rb' ) as f:
    data = pickle.load( f )

set_blocks_env()
set_experiment_env()


########## ANALYSIS ################################################################################



""" ##### [?] Did the recording work? ##################################### """
# Print all metadata
for i, datum in enumerate( data ):
    print( f"{datum['t']}:{i}: {datum['msg']}, {type(datum['data'])}" )



""" [?] How stable are the symbols? Do the memories they are derived from always change? """

symbolStability = list()
for i, datum in enumerate( data ):
    if datum['msg'] == 'symbols':
        sym = dict()
        for pair in datum['data']:
            sym[ pair['symbol'].label ] = { 
                'pose'   : extract_pose_as_homog( pair['symbol'] ),
                'parIdx' : pair['parent'].index,
                'parS'   : pair['parent'].score,
            }
        symbolStability.append( sym )


""" [?] Do the readings that become symbols get erased after the action? Why? """
