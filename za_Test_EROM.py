########## INIT ####################################################################################

### Standard ###
import os
from time import sleep
from random import random

### Special ###
import numpy as np

### ASPIRE ###
from aspire.symbols import ObjPose, GraspObj, extract_pose_as_homog
from aspire.env_config import env_var
from aspire.utils import get_confusion_matx, match_name, roll_outcome

from aspire.env_config import set_blocks_env
from TaskPlanner import set_experiment_env
set_blocks_env()
set_experiment_env()

### Local ###
from EROM import EROM



########## EROM TEST HARNESS #######################################################################

_STEP_SLEEP_S = 0.5


def expand_pose_from_vec( vec ):
    """ Insert position into the I pose """
    rtnPose = np.eye(4)
    rtnPose[0:3,3] = vec
    return rtnPose


def select_confusion_row( confMatx, label ):
    """ Get the appropriate row of the confusion matrix """
    i = env_var("_BLOCK_NAMES").index( match_name( label ) )
    return confMatx[i,:]


def get_confused_dist( confMatx, label, shorten = True ):
    """ Get the confused distribution for the label """
    row   = select_confusion_row( confMatx, label )
    names = env_var("_BLOCK_NAMES")
    dist  = dict()
    for i in range( env_var("_N_CLASSES") ):
        nam_i = names[i][:3] if shorten else names[i]
        dist[ nam_i ] = row[i]
    return dist


class FakeReadingsGen:
    """ Generate readings that let me know if EROM is working """

    def __init__( self, rareRate : float, noiseRad_m : float, commonLoci : list[dict], rareLoci : list[dict] ):
        """ Setup generation """
        self.rRate  = rareRate
        self.nsRad  = noiseRad_m
        self.cnfMtx = get_confusion_matx( env_var("_N_CLASSES"), confuseProb = env_var("_CONFUSE_PROB") )
        self.common : list[GraspObj] = list()
        self.rare   : list[GraspObj] = list()
        
        for locus in commonLoci:
            self.common.append( GraspObj(
                label = locus['label'],
                pose  = ObjPose( expand_pose_from_vec( locus['posn'] ) ),
            ) )
        for locus in rareLoci:
            self.rare.append( GraspObj(
                label = locus['label'],
                pose  = ObjPose( expand_pose_from_vec( locus['posn'] ) ),
            ) )


    def perturb_pose( self, pose : np.ndarray ):
        """ Jiggle pose """
        rtnPose = pose.copy()
        for i in range(3):
            rtnPose[i,3] += -self.nsRad + random()*2.0*self.nsRad
        return rtnPose
    

    def roll_label_dist( self, inLabel ):
        """ Jiggle label """
        odds   = get_confused_dist( self.cnfMtx, inLabel )
        outLbl = roll_outcome( odds )
        return get_confused_dist( self.cnfMtx, outLbl )


    def generate( self ):
        """ Generate a list of noisy readigs """
        i = 0
        r = dict()

        def roll( locus : GraspObj ):
            """ Generate a reading from the `locus` """
            nonlocal i, r
            i += 1
            n = f"obj{i}"
            r[n] = {
                "Pose"        : self.perturb_pose( extract_pose_as_homog( locus ) ),
                'Probability' : self.roll_label_dist( locus.label )
            }

        for locus in self.common:
            roll( locus )
        for locus in self.rare:
            if random() <= self.rRate:
                roll( locus )

        return r

if __name__ == "__main__":
    from pprint import pprint
    

    frg = FakeReadingsGen(
        rareRate = 1.0/4.0,
        noiseRad_m = 0.5 * env_var("_BLOCK_SCALE"),
        commonLoci = [{ 'label': 'grn', 'posn': [0,1,0] },{ 'label': 'blu', 'posn': [1,1,0] },],
        rareLoci   = [{ 'label': 'ylw', 'posn': [1,1,0] },],
    )

    for i in range(3):
        pprint( frg.generate() )

    os.system( 'kill %d' % os.getpid() ) 