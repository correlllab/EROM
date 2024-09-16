########## INIT ####################################################################################
import os

########## BELIEFS #################################################################################

def set_belief_env():
    os.environ["_CONFUSE_PROB"]     = 0.025 # 0.05 # 0.001 # 0.001 # 0.025 # 0.05
    os.environ["_NULL_THRESH"]      = 0.75
    os.environ["_EXIST_THRESH"]     = 0.05
    os.environ["_MAX_UPDATE_RAD_M"] = 2.0*os.environ["_BLOCK_SCALE"]
    os.environ["_NULL_EVIDENCE"]    = True


########## OBJECT PERMANENCE #######################################################################

def set_EROM_env():
    os.environ["_SCORE_FILTER_EXP"]  =   0.75 # During a belief update, accept the new score at this rate
    os.environ["_SCORE_DECAY_TAU_S"] =  20.0 # Score time constant, for freshness 
    os.environ["_SCORE_DIV_FAIL"]    = 2.0
    os.environ["_OBJ_TIMEOUT_S"]     = 120.0 # Readings older than this are not considered
    os.environ["_UPDATE_PERIOD_S"]   =   4.0 # Number of seconds between belief updates
    os.environ["_DEF_NULL_SCORE"]    =   0.75 # Default null score if there was no comparison
    os.environ["_LKG_SEP"]           = 0.80*os.environ["_BLOCK_SCALE"] # 0.40 # 0.60 # 0.70 # 0.75
    os.environ["_CUT_MERGE_S_FRAC"]  = 0.325 # 0.325 # 0.40 # 0.20
    os.environ["_CUT_SCORE_FRAC"]    = 0.25
    os.environ["_REIFY_SUPER_BEL"]   = 1.01


    ##### PLANNER #########################################################

    os.environ["_N_XTRA_SPOTS"]  = 4
    os.environ["_CHANGE_THRESH"] = 0.40
    os.environ["_BT_LOOK_DIV"]   = 0.5