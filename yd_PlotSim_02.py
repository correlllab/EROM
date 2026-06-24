########## INIT ####################################################################################
import pickle

from collections import deque
from typing import Deque

from analysis_Utils import Loc, make_histo, crash_out
from yb_SimpleSim_02 import StepRecord, SimBlock, Action


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


        ##### Load Data ###################################################

        simData : Deque[Deque[StepRecord]] = simRes[ setNam ][ test ]

        
        ##### Analyze Data ################################################

        ##### Makespan (Steps) ###################
        msDqu_stp : Deque[int] = deque()

        for episode in simData:
            ms_i = 0.0
            for datum in episode: 
                ms_i += 1
            msDqu_stp.append( ms_i )

        make_histo( msDqu_stp, 
                    f"{setNam}:{test}: Makespan [Steps]", 
                    f"{Loc._PLOT_DIR}{setNam}_{test}_Makespan-Step{Loc.plotExt}", 
                    xLabel = 'Makespan', yLabel = 'Occurrences', forceYlim = False, savefig = True )


        ##### Makespan (Seconds) ################# 
        msDqu_sec  : Deque[float] = deque()

        for episode in simData:
            ms_i = 0.0
            for datum in episode: 
                ms_i += (datum.tSearch + datum.tAction)
            msDqu_sec.append( ms_i )

        

########## EXIT ####################################################################################
crash_out( False )