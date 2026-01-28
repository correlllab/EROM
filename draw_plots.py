########## PLOTTING FUNCTIONS ######################################################################
import numpy as np
import matplotlib.pyplot as plt


_TITLE_FONT_SIZE = 13
_N_TRIALS        = 20
_TIGHT_MARGIN    =  0.05


def make_histo( series, plotTitle, fName, xLabel = 'Makespan', yLabel = 'Occurrences', forceYlim = True, savefig = True ):
    """ Create Histogram """
    if savefig:
        plt.clf()
    plt.margins( _TIGHT_MARGIN )
    print( f"\n{plotTitle}" )
    print( f"Mean: ___ {np.mean(series)}" )
    print( f"Median: _ {np.median(series)}" )
    print( f"Std.Dev.: {np.std(series)}" )
    plt.hist( series )
    plt.title( plotTitle, fontsize = _TITLE_FONT_SIZE ) # Set the title && font size
    plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    if forceYlim:
        plt.ylim( (0, _N_TRIALS,) )
    plt.tight_layout()
    if savefig:
        plt.savefig( fName )
        return plt.gca()


def make_multi_histo( multiSeries, seriesNames, plotTitle = None, fName = "output.pdf", xLabel = None, yLabel = None, 
                      forceYlim = True, savefig = True, titleFontSize_pt = _TITLE_FONT_SIZE ):
    """ Create Histogram across Categories on the same Axes """
    if savefig:
        plt.clf()
    plt.margins( _TIGHT_MARGIN )
    print( f"\n### {plotTitle} ###" )
    for i, series in enumerate( multiSeries ):
        print( f"\t{seriesNames[i]}" )
        print( f"\tMean: ___ {np.mean(series)}" )
        print( f"\tMedian: _ {np.median(series)}" )
        print( f"\tStd.Dev.: {np.std(series)}" )
    plt.hist( multiSeries, label = seriesNames )
    if plotTitle is not None:
        plt.title( plotTitle, fontsize = titleFontSize_pt ) # Set the title && font size
    if xLabel is not None:
        plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    if yLabel is not None:
        plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    if savefig:
        plt.legend( loc = 'upper right' )
    if forceYlim:
        plt.ylim( (0, _N_TRIALS,) )
    plt.tight_layout()
    if savefig:
        plt.savefig( fName )
        return plt.gca()