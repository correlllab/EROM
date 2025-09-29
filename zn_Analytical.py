import os
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

"""
SIMPLIFCIATIONS:
* Use mean times for {`mv`, `pl`, `rm`} 
"""

_TITLE_FONT_SIZE = 13
_TIGHT_MARGIN    =  0.05

def make_line_plot( X, Y, plotTitle, fName = None, xLabel = 'Makespan', yLabel = 'Occurrences', savefig = True ):
    """ Create Histogram """
    if savefig:
        plt.clf()
    plt.margins( _TIGHT_MARGIN )
    print( f"\n{plotTitle}" )
    # plt.hist( series )
    plt.plot( X, Y )
    plt.title( plotTitle, fontsize = _TITLE_FONT_SIZE ) # Set the title && font size
    plt.xlabel( xLabel ) # ---------------- Setting the x-axis label
    plt.ylabel( yLabel ) # ---------------- Setting the y-axis label
    plt.tight_layout()
    if savefig:
        plt.savefig( fName )
        return plt.gca()
    else:
        plt.show()


def m_An( m_mv, m_pl, m_rm, PofN_A ) -> float:
    """ Equation 2: Makespan contribution to place Block A, Closed form """
    return (m_mv + m_pl + PofN_A * m_rm) / (1.0 - PofN_A)


def m_i( i : int, m_mv : float, m_pl : float, PofN : list[float], m_rm : float ) -> float:
    """ Equation 5: Makespan contribution to place Block i, Closed form """

    def alpha( i, m_mv, m_pl, PofN, m_rm ):
        return m_mv + m_pl + PofN[i] * m_rm
    
    def beta( i, m_rm, m_ : list[float] = None  ):
        rtnBta = m_rm
        for k in range( i-1 ):
            rtnBta += m_rm + m_[k]
        return rtnBta

    m_i = [0.0,]
    
    for ii in range( 1, i+1 ):
        if ii == 1:
             m_i.append( m_An( m_mv, m_pl, m_rm, PofN[1] ) )
        else:
            a  = alpha( ii, m_mv, m_pl, PofN, m_rm )
            n2 = 0.0
            for j in range( 1, ii-1 ):
                n2 += PofN[j] * beta( j, m_rm, m_i )
            dn = 1 - PofN[ii]
            for j in range( 1, ii-1 ):
                dn -= PofN[j]
            m_i.append( (a + n2)/dn )

    return sum( m_i )


########## MAIN ####################################################################################
if __name__ == "__main__":
    tAct = 30.00
    m_mv = tAct
    m_pl = tAct
    m_rm = tAct
    # Pcas = np.linspace( 0.0, 0.999, num = 100, endpoint = True )
    # Pcas = np.linspace( 0.0, 0.499, num = 100, endpoint = True )
    # Pcas = np.linspace( 0.0, 0.400, num = 100, endpoint = True )
    Pcas = np.linspace( 0.0, 0.450, num = 100, endpoint = True )
    Ymak = deque()
    for Pcnf in Pcas:
        PofN = [None, Pcnf, Pcnf, Pcnf,]
        m_A  = m_An( m_mv, m_pl, m_rm, PofN[1] )
        m_1  = m_i( 1, m_mv, m_pl, PofN, m_rm )
        m_3  = m_i( 3, m_mv, m_pl, PofN, m_rm )
        print( f"1-Tower Makespan: {m_1} -vs- {m_A}" )
        print( f"3-Tower Makespan: {m_3}\n" )
        Ymak.append( m_3 )
    make_line_plot( Pcas, Ymak, "Test Plot",
                   xLabel = 'Confusion', yLabel = "Makespan [s]",
                   savefig = False )


########## EXIT ####################################################################################
print( "\n\n" )
os.system( 'kill %d' % os.getpid() ) 


    