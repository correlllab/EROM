########## INIT ####################################################################################
import pickle, os, gc, traceback, json

from collections import deque
from typing import Any, Deque
from pprint import pprint
from random import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from aspire.symbols import GraspObj, ObjPose, extract_position, euclidean_distance_between_symbols
from aspire.BlocksTask import set_blocks_env
from aspire.env_config import env_var
from TaskPlanner import set_experiment_env
from draw_beliefs import set_render_env
from magpie_control.realsense_wrapper import MPCD
from Reader import EROM_Reader

from homog_utils import homog_xform, diff_mag

##### Environment && Constants ############################################
set_blocks_env()
set_experiment_env()
set_render_env()

########## CONSTANTS ###############################################################################

_DATA_DRIVE = "STARGAZER/DATA_TANK"

_PLOT_DIR   = "/media/james/FILEPILE/EROM/data/plots/"
_GC_CYCLE   = False 
_F_EXTRACT  = f"{_PLOT_DIR}outData.pkl"
_T_EXTRACT  = f"{_PLOT_DIR}outText.json"

_MIN_STATE_SIZE_BYTES = 500.0

_TITLE_FONT_SIZE = 13
_TIGHT_MARGIN    =  0.05

_DEFAULT_DIV = 100 #80 #100 #200

tests = [
    "KC-KP",
    "SC-KP",
    "KC-SP",
    "SC-SP",
]

longTestNames = [
    "Known Class & Known Pose", 
    "Sensed Class & Known Pose", 
    "Known Class & Sensed Pose", 
    "Sensed Class & Sensed Pose", 
]

datasets = [
    [ f"/media/james/{_DATA_DRIVE}/2025-08B_{test}" for test in tests ],
    [ f"/media/james/{_DATA_DRIVE}/RWB_2025-09_{test}" for test in tests ],
]

dataLabels = ["RGB", "RBW",]
datNamLong = {
    "RGB": "Red-Green-Blue", 
    "RBW": "Red-Black-White",
}

blcNam = {
    "RGB": ['redBlock','grnBlock','bluBlock',], 
    "RBW": ['redBlock','blkBlock','whtBlock',],
}

eBlcNam = {
    "RGB": ['redBlock', 'grnBlock', 'bluBlock', env_var("_NULL_NAME"),], 
    "RBW": ['redBlock', 'blkBlock', 'whtBlock', env_var("_NULL_NAME"),],
}

plotExt = ".pdf"

