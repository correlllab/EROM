########## INIT ####################################################################################
import pickle, os, traceback, json
from collections import deque
from copy import deepcopy
from pprint import pprint
from typing import Deque

import numpy as np

from aspire.env_config import env_var
from aspire.symbols import GraspObj, ObjPose, euclidean_distance_between_symbols
from aspire.BlocksTask import set_blocks_env

from TaskPlanner import set_experiment_env
from draw_beliefs import set_render_env
from utils import deep_copy_memory_list

##### Environment && Constants ############################################
set_blocks_env()
set_experiment_env()
set_render_env()


########## SETUP ###################################################################################
_JSON_PATH  = "data/allData.txt"
_DATA_DRIVE = "DATA_TANK"
# _PLOT_DIR   = "/media/james/FILEPILE/EROM/data/plots/"
_PLOT_DIR   = "data/plots/"

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

# paths = [ f"/media/james/{_DATA_DRIVE}/2025-08_{test}" for test in tests ]
datasets = [
    [ f"/media/james/{_DATA_DRIVE}/2025-08B_{test}" for test in tests ],
    [ f"/media/james/{_DATA_DRIVE}/RWB_2025-09_{test}" for test in tests ],
]

dataLabels = ["RGB", "RBW",]
datNamLong = {
    "RGB": "Red-Green-Blue", 
    "RBW": "Red-Black-White",
}

fNames = [ f"{_PLOT_DIR}{test}" for test in tests  ]

plotExt = ".pdf"


########## SAVE: DATA PROCESSING ###################################################################
_SAVE_DATA = True
_LOAD_DATA = True 