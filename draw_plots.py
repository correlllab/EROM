########## PLOTTING FUNCTIONS ######################################################################
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


_TITLE_FONT_SIZE = 13
_N_TRIALS        = 20
_TIGHT_MARGIN    =  0.05


def init_tight_fig( savefig = True ):
    """ Common setup tasks across plots """
    if savefig:
        plt.clf()
    plt.margins( _TIGHT_MARGIN )


def figure_data_report( multiSeries, seriesNames = None, plotTitle = None ):
    if plotTitle is not None:
        print( f"\n### {plotTitle} ###" )
    if seriesNames is None:
        print( f"Mean: ___ {np.mean(multiSeries)}" )
        print( f"Median: _ {np.median(multiSeries)}" )
        print( f"Std.Dev.: {np.std(multiSeries)}" )
    elif len( seriesNames ) == len( multiSeries ):
        for i, series in enumerate( multiSeries ):
            print( f"\t{seriesNames[i]}" )
            print( f"\tMean: ___ {np.mean(series)}" )
            print( f"\tMedian: _ {np.median(series)}" )
            print( f"\tStd.Dev.: {np.std(series)}" )


def make_histo( series, plotTitle, fName, xLabel = 'Makespan', yLabel = 'Occurrences', forceYlim = True, savefig = True ):
    """ Create Histogram """
    init_tight_fig( savefig )
    figure_data_report( series, plotTitle = plotTitle )
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
    else:
        plt.show()


def make_multi_histo( multiSeries, seriesNames, plotTitle = None, fName = "output.pdf", xLabel = None, yLabel = None, 
                      forceYlim = True, savefig = True, titleFontSize_pt = _TITLE_FONT_SIZE, decimals = 3, wholeStep = 5 ):
    """ Create Histogram across Categories on the same Axes """
    init_tight_fig( savefig )
    figure_data_report( multiSeries, seriesNames, plotTitle )

    if decimals == 0:
        data = deque()
        for series in multiSeries:
            data.extend( series )
        sMin = int( min( data ) )
        sMax = int( max( data ) )
        bins = deque([sMin,])
        while (bins[-1] + wholeStep) < sMax:
            bins.append( bins[-1] + wholeStep )
        if bins[-1] < sMax:
            bins.append( bins[-1] + wholeStep )

        _, bin_edges, _ = plt.hist( multiSeries, label = seriesNames, bins = bins )
    else:
        _, bin_edges, _ = plt.hist( multiSeries, label = seriesNames )

    # bin_ticks = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    if decimals > 0:
        bin_ticks = [float(f"{item:.{decimals}f}") for item in bin_edges[:-1]]
    else:
        bMin = int( min( bin_edges ) )
        bMax = int( max( bin_edges ) )
        bTix = deque()
        bTix.append( bMin )
        while (bTix[-1] + wholeStep) < bMax:
            bTix.append( bTix[-1] + wholeStep )
        if bTix[-1] < bMax:
            bTix.append( bTix[-1] + wholeStep )
        # bin_ticks = [float(f"{item:.{decimals}f}") for item in bin_edges[:-1]]
        bin_ticks = list( bTix )
        # bin_ticks = [int(float(f"{item}")) for item in bin_edges[:-1]]
    if plotTitle is not None:
        plt.title( plotTitle, fontsize = titleFontSize_pt ) # Set the title && font size
    if xLabel is not None:
        plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
        # Set the x-axis tick positions to the midpoints
        plt.xticks( bin_ticks )
    if yLabel is not None:
        plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    if savefig:
        plt.legend( loc = 'upper right' )
    if forceYlim:
        plt.ylim( (0, _N_TRIALS,) )
        plt.yticks( list( range(0, _N_TRIALS+1, 2) ) )
    plt.tight_layout()
    if savefig:
        plt.savefig( fName )
        return plt.gca()
    else:
        plt.show()
    

def make_whisker( multiSeries, seriesNames, plotTitle = None, fName = "output.pdf", yLabel = None, 
                  forceYlim = True, savefig = True, titleFontSize_pt = _TITLE_FONT_SIZE, outliers = True ):
    plt.clf()
    init_tight_fig( savefig )
    figure_data_report( multiSeries, seriesNames, plotTitle )
    # Create the plot
    plt.boxplot( x          = multiSeries,
                 labels     = seriesNames,
                 showfliers = outliers   )
    if plotTitle is not None:
        plt.title( plotTitle, fontsize = titleFontSize_pt ) # Set the title && font size
    if yLabel is not None:
        plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    if forceYlim:
        plt.ylim( (0, _N_TRIALS,) )
    plt.tight_layout()
    if savefig:
        plt.savefig( fName )
        return plt.gca()
    else:
        plt.show()