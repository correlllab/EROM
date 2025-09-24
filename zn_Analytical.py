import numpy as np

"""
SIMPLIFCIATIONS:
* Use mean times for {`mv`, `pl`, `rm`} 
"""


def m_An( m_mv, m_pl, m_rm, PofN_A ) -> float:
    """ Equation 2: Makespan contribution to place Block A, Closed form """
    return (m_mv + m_pl + PofN_A * m_rm) / (1.0 - PofN_A)


def m_i( i : int, m_mv : float, m_pl : float, PofN : list[float], m_rm : float ) -> float:
    """ Equation 5: Makespan contribution to place Block i, Closed form """

    def alpha( i, m_mv, m_pl, PofN, m_rm ):
        return m_mv + m_pl + PofN[i] * m_rm
    
    def beta( m_rm, m_ : list[float]  ):
        if len( m_ ) > 1:
            rtnBta = m_rm
            for k in range( len( m_ ) ):
                rtnBta += m_rm + m_[k]
            return rtnBta
        elif len( m_ ) == 1:
            return m_rm + m_[0]
        else:
            raise ValueError( f"Got a list of length {len( m_ )}" )
    
    a = alpha( i, m_mv, m_pl, PofN, m_rm )