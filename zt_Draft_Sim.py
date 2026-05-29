"""
zt_Sim_Draft.py
Draft sim
"""

from dataclasses import dataclass, field 


@dataclass
class Block:
    """ Symbol of a physical object """
    posErr : float = 0.0
    actCls : str   = "INVALID"
    snsCls : str   = "INVALID"


class Event_Recorder:
    """ Record events and make it easy to take metrics """
    pass


class Dice_Engine:
    """ Use the recorded metrics to generate outcomes """
    pass


class Robot_Planner:
    """ Use the recorded metrics to simulate actions """
    pass


