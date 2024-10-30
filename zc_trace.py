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


########## HELPER FUNCTIONS ########################################################################

def get_symbol_parents( objLst : list[GraspObj] ):
    """ Filter out the objects that are parents of symbols """
    rtnLst = list()
    for obj in objLst:
        if obj.SYM:
            rtnLst.append( obj )
    return rtnLst


########## ANALYSIS ################################################################################



""" ##### [?] Did the recording work? ##################################### """
# Print all metadata
for i, datum in enumerate( data ):
    print( f"{datum['t']}:{i}: {datum['msg']}, {type(datum['data'])}" )



""" [?] How stable are the symbols? Do the memories they are derived from always change? """
""" [?] Do the symbol probabilities trend upward? """
""" [?] Do the symbol quality scores trend upward? """
""" [?] What is the impact of artifically boosting the quality of moved readings? Did it help? """
# FIXME: PLOTS FOR ABOVE QUESTIONS

symbolStability = list()
combosPerAction = list()

for i, datum in enumerate( data ):
    if datum['msg'] == 'symbols':

        sym = { 't': datum['t'] }
        for pair in datum['data']['pairs']:
            sym[ pair['symbol'].label ] = { 
                'prob'   : pair['symbol'].prob,
                'score'  : pair['symbol'].score,
                'pose'   : extract_pose_as_homog( pair['symbol'] ),
                'parIdx' : pair['parent'].index,
                'parS'   : pair['parent'].score,
            }
        symbolStability.append( sym )

        combosPerAction.append({ 
            't'      : datum['t'],
            'combos' : datum['data']['combos'],
        })



""" [?] Do the readings that become symbols get erased after the action? Why? """
# FIXME: ANALYSIS FOR ABOVE QUESTION
symbolDeletions = list()
for i, datum in enumerate( data ):
    symDel = { 't': datum['t'] }
    added  = False
    if datum['msg'] == 'ranking':
        symDel['rnkRct'] = datum['data']["remRec"]
        symDel['rnkCut'] = datum['data']["remCut"]
        added = True
    if datum['msg'] == 'memory':
        symDel['LKGrec'] = datum['data']["LKGrec"]
        symDel['LKGcut'] = datum['data']["LKGcut"]
        added = True
    if added:
        symbolDeletions.append( symDel )


