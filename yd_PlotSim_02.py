########## INIT ####################################################################################
import pickle

from collections import deque
from typing import Deque

from analysis_Utils import Loc, make_histo, crash_out
from yb_SimpleSim_02 import StepRecord, SimBlock, Action
from draw_plots import make_whisker


########## MAIN ####################################################################################
simRes  = dict()
setData = dict()

with open( Loc._SIM_DATA_PATH, 'rb' ) as f:
    simRes = pickle.load(f)

### For every block set ###
for iii, paths in enumerate( Loc.datasets ):

    setNam   = Loc.dataLabels[iii]
    classes  = Loc.blcNam[setNam]
    eClasses = Loc.eBlcNam[setNam]

    setData[ setNam ] = {
        "msStep"  : dict(),            
        "msTime"  : dict(),        
    }

    ### For every scenario ###
    for ii, test in enumerate( Loc.tests ):


        ##### Load Data ###################################################

        simData : Deque[Deque[StepRecord]] = simRes[ setNam ][ test ]

        
        ##### Analyze Data ################################################

        ##### Makespan (Steps) ###################
        msDqu_stp : Deque[int]   = deque()

        for episode in simData:
            ms_i = 0.0
            for datum in episode: 
                ms_i += 1
            msDqu_stp.append( ms_i )

        setData[ setNam ]["msStep"][ test ] = deque( msDqu_stp )

        make_histo( msDqu_stp, 
                    f"{setNam}:{test}: Makespan [Steps]", 
                    f"{Loc._PLOT_DIR}{setNam}_{test}_Makespan-Step{Loc.plotExt}", 
                    xLabel = 'Makespan', yLabel = 'Occurrences', forceYlim = False, savefig = True )


        ##### Makespan (Seconds) ################# 
        msDqu_sec  : Deque[float] = deque()

        for episode in simData:
            ms_i = 0.0
            for datum in episode: 
                print( datum.tSearch, datum.tAction )
                ms_i += (datum.tSearch + datum.tAction)
            msDqu_sec.append( ms_i )

        setData[ setNam ]["msTime"][ test ] = deque( msDqu_sec )
        
    series = deque()
    sNames = deque()

    ### For every scenario ###
    for ii, test in enumerate( Loc.tests ):
        series.append( list( setData[ setNam ]["msStep"][ test ] ) )
        sNames.append( test )

    make_whisker( series, sNames, plotTitle = f"{setNam} Makespan [steps]", fName = "output.pdf", yLabel = "Steps", 
                  forceYlim = False, savefig = False, outliers = True )
    

    series = deque()
    sNames = deque()

    ### For every scenario ###
    for ii, test in enumerate( Loc.tests ):
        series.append( list( setData[ setNam ]["msTime"][ test ] ) )
        sNames.append( test )

    make_whisker( series, sNames, plotTitle = f"{setNam} Makespan [s]", fName = "output.pdf", yLabel = "Seconds", 
                  forceYlim = False, savefig = False, outliers = True )
        

########## EXIT ####################################################################################
crash_out( False )