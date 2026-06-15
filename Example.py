from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Deque
from collections import deque

from aspire.symbols import GraspObj, ObjPose, extract_position, euclidean_distance_between_symbols


class PlanStatus( Enum ):
    INVALID     = 0
    NOT_PLANNED = 1
    PLANNED_OK  = 2
    PLANNED_BAD = 3
    TASK_DONE   = 4


class ActionStatus( Enum ):
    INVALID     = 0
    NO_ACTION   = 1
    SUCCESS     = 2
    FAILURE     = 3


@dataclass
class Example:
    
    ## State ##
    sensBeliefs : Deque[GraspObj]    = field( default_factory = deque )
    trueReadings: Deque[GraspObj]    = field( default_factory = deque )
    sensSymbols : Deque[GraspObj]    = field( default_factory = deque )
    trueSymbols : dict[str,GraspObj] = field( default_factory = dict  )

    ## Address ##
    dataset: str = ""
    test   : str = ""
    episode: int = -1
    step   : int = -1

    ## Perception ##
    p_AllBlocks : bool            = False
    missingBlock: list[str]       = field( default_factory = list )
    p_Confused  : bool            = False
    confusedBloc: list            = field( default_factory = list )
    positionErrs: dict[str,float] = field( default_factory = dict )
    N_halluc    : int             = 0
    
    ## Planning ##
    planned : PlanStatus = PlanStatus.INVALID

    ## Action ##
    action : ActionStatus = ActionStatus.INVALID

    ## Next State ##
    nextState : Example = None
