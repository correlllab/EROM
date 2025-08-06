########## INIT && LOAD DATA #######################################################################
import pickle, os
from collections import deque

import numpy as np

from aspire.symbols import GraspObj
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import set_experiment_env
from draw_beliefs import ( set_render_env, render_memory_list, scan_geo, vispy_geo_list_window, cpcd_geo )

# path = "data/Baseline_2025-03-11"
# path = "data/Baseline_2025-03-13" 
# path = "data/Baseline_2025-04-28" 
# path =  

_PLOT_DIR = "data/plots/"

if 0:
    path      = "data/Pose-Help_Baseline_2025-05-20"
    plotTitle = "Makespan Distribution with Pose Help"
    fName     = f"{_PLOT_DIR}poseHelp.pdf"

if 1:
    path      = "data/Basic-Baseline_2025-05-08"
    plotTitle = "Makespan Distribution with Pose & Class Help"
    fName     = f"{_PLOT_DIR}classPoseHelp.pdf"

pkls = [os.path.join( path, item ) for item in os.listdir( path ) if ".pkl" in f"{item}".lower()]
for pkl in pkls:
    print( pkl )



##### Environment && Constants ############################################

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
from pprint import pprint

##### Success Rate ########################################################
N   = 0
S   = 0
F   = 0
MTS = 0.0
MTF = 0.0

result = {
    'Ntrial' : 0,
    'outcome': list(),
    'tRun'   : list(),
    'sRun'   : list(),
}

print( f"\nThere are {len(pkls)} files to analyze:\n" )

for pkl in pkls:
    print( f"About to open {pkl} ..." )
    with open( pkl, 'rb' ) as f:
        N += 1
        print( f"Count: {N}" )
        data = pickle.load( f )
        tRun = data[-1]['t'] - data[0]['t']

        result['Ntrial'] += 1
        result['tRun'  ].append( tRun )

        end  =  False
        for i in range(1,11):
            msg  = data[-i]['msg']
            print( msg )
            if "Status.FAILURE" in msg:
                F += 1
                print( "FAILURE" )
                MTF += tRun
                end = True
                result['outcome'].append( 0 )
                break
            elif "Status.SUCCESS" in msg:
                S += 1
                print( "SUCCESS" )
                MTS += tRun
                end = True
                result['outcome'].append( 1 )
                break
        if not end:
            F += 1
            print( "FAILURE" )
            MTF += tRun
            result['outcome'].append( -1 ) # Indeterminate should be counted as a failure
        print()

if S > 0:
    MTS /= S
if F > 0:
    MTF /= F
print( f"\n{N} episodes, Success Rate: {S*1.0/N}, Failure Rate: {F*1.0/N}, Sanity Check == 0.0: {1.0-S*1.0/N-F*1.0/N}" )
print( f"Mean Time to Success: {[int(item) for item in divmod( MTS, 60.0 )]}, Mean Time to Failure: {[int(item) for item in divmod( MTF, 60.0 )]}" )

pprint( result )

import matplotlib.pyplot as plt



print()
print( f"Mean: ___ {np.mean(result['tRun'])} [s]" )
print( f"Median: _ {np.median(result['tRun'])} [s]" )
print( f"Std.Dev.: {np.std(result['tRun'])} [s]" )

plt.hist( result['tRun'] )
plt.title( plotTitle ) # Set the title
plt.xlabel('Makespan [s]')  # Setting the x-axis label
plt.ylabel('Occurrences') # Setting the y-axis label
plt.savefig( fName )
plt.show() 


########## DRAW SCANS ##############################################################################

totMem  = list()
camPose = np.eye(4)

_FETCH_CAM = False
_DRAW_SYMB = False
_DRAW_PCDS = False

if (_DRAW_SYMB or _DRAW_PCDS):

    dPth = pkls[-2]

    print( f"About to open {dPth} ..." )

    data = list()
    with open( dPth, 'rb' ) as f:
        data = pickle.load( f )

    for i, datum in enumerate( data ):

        if ((_FETCH_CAM or _DRAW_PCDS) and (datum['msg'] == 'camera')):
            camPose = datum['data'].copy()
            # print( datum['data'].keys() )

        if _DRAW_PCDS and (datum['msg'] == 'memory'):
            readings = datum['data']['scan']
            totGeo   = deque()
            for rdg in readings:
                if len( rdg.cpcd ) > 20:
                    totGeo.extend( cpcd_geo( rdg ) )
                    totGeo.extend( scan_geo( rdg ) )

            # print( type( totGeo ) )
            vispy_geo_list_window( list( totGeo ) )


            
        if _DRAW_SYMB and (datum['msg'] == 'symbols'):
            # render_memory_list( syms = datum['data']['scan'] )
            render_memory_list( syms = datum['data'] )



print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 