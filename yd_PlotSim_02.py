########## INIT ####################################################################################
import pickle

from collections import deque

from analysis_Utils import Loc
from yb_SimpleSim_02 import StepRecord


########## MAIN ####################################################################################
simRes = dict()

with open( Loc._SIM_DATA_PATH, 'rb' ) as f:
    simRes = pickle.load(f)

### For every block set ###
for iii, paths in enumerate( Loc.datasets ):

    setNam   = Loc.dataLabels[iii]
    classes  = Loc.blcNam[setNam]
    eClasses = Loc.eBlcNam[setNam]

    ### For every scenario ###
    for ii, test in enumerate( Loc.tests ):

        simDict = simRes[ setNam ][ test ]
        simData = deque()

        for episode in simDict:
            epDqu = deque()
            for datum in episode:
                # epDqu.append( StepRecord( **datum ) )
                epDqu.append( datum )
            simData.append( epDqu )
