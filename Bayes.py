########## INIT ####################################################################################

###### Imports ######

### Standard ###
import time
now = time.time

### Special ###
import numpy as np

### MAGPIE Ctrl ###
from magpie_control.poses import repair_pose, vec_unit
from magpie_control.homog_utils import R_x, R_y, posn_from_xform

### ASPIRE ###
from aspire.env_config import env_var
from aspire.utils import ( extract_dct_values_in_order, sorted_obj_labels, multiclass_Bayesian_belief_update, 
                           get_confusion_matx, get_confused_class_reading )
from aspire.symbols import ( euclidean_distance_between_symbols, extract_pose_as_homog, 
                             p_symbol_inside_workspace_bounds, ObjPose, GraspObj )


### Local ###
from utils import set_quality_score



########## HELPER FUNCTIONS ########################################################################

def extract_class_dist_in_order( obj, order = env_var("_BLOCK_NAMES"), insertZero = False ):
    """ Get the discrete class distribution, in order according to environment variable """
    if isinstance( obj, dict ):
        return np.array( extract_dct_values_in_order( obj, order, insertZero = insertZero ) )
    else:
        return np.array( extract_dct_values_in_order( obj.labels, order, insertZero = insertZero ) )


def extract_class_dist_sorted_by_key( obj, insertZero = False ):
    """ Get the discrete class distribution, sorted by key name """
    return np.array( extract_dct_values_in_order( obj.labels, sorted_obj_labels( obj ), insertZero = insertZero ) )


def extract_class_dist_in_order( obj, order = env_var("_BLOCK_NAMES"), insertZero = False ):
    """ Get the discrete class distribution, in order according to environment variable """
    if isinstance( obj, dict ):
        return np.array( extract_dct_values_in_order( obj, order, insertZero = insertZero ) )
    else:
        return np.array( extract_dct_values_in_order( obj.labels, order, insertZero = insertZero ) )


def exp_filter( lastVal, nextVal, rate01 ):
    """ Blend `lastVal` with `nextVal` at the specified `rate` (Exponential Filter) """
    assert 0.0 <= rate01 <= 1.0, f"Exponential filter rate must be on [0,1], Got {rate01}"
    return (nextVal * rate01) + (lastVal * (1.0 - rate01))


def p_sphere_inside_plane_list( qCen, qRad, planeList ):
    """ Return True if a sphere with `qCen` and `qRad` can be found above every plane in `planeList` = [ ..., [point, normal], ... ] """
    if len( qCen ) == 4:
        qCen = posn_from_xform( qCen )
    for (pnt_i, nrm_i) in planeList:
        # print(pnt_i, nrm_i)
        dif_i = np.subtract( qCen, pnt_i )
        dst_i = np.dot( dif_i, vec_unit( nrm_i ) )
        # print( f"Distance to Plane: {dst_i}" )
        if dst_i < qRad:
            return False
    return True

def get_D405_FOV_frustum( camXform ):
    """ Get 5 <point, normal> pairs for planes bounding an Intel RealSense D405 field of view with its focal point at `camXform` """
    ## Fetch Components ##
    rtnFOV   = list()
    camXform = repair_pose( camXform ) # Make sure all bases are unit vectors
    camRot   = camXform[0:3,0:3]
    cFocus   = camXform[0:3,3]
    ## Depth Limit ##
    dNrm = camRot.dot( [0.0,0.0,-1.0,] )
    dPos = np.eye(4)
    dPos[2,3] = env_var("_D405_FOV_D_M")
    dPnt = camXform.dot( dPos )[0:3,3]
    rtnFOV.append( [dPnt, dNrm,] )
    ## Top Limit ##
    tNrm = camRot.dot( R_x( -np.radians( env_var("_D405_FOV_V_DEG")/2.0 ) ).dot( [0.0,-1.0,0.0] ) )
    tPnt = cFocus.copy()
    rtnFOV.append( [tPnt, tNrm,] )
    ## Bottom Limit ##
    bNrm = camRot.dot( R_x( np.radians( env_var("_D405_FOV_V_DEG")/2.0 ) ).dot( [0.0,1.0,0.0] ) )
    bPnt = cFocus.copy()
    rtnFOV.append( [bPnt, bNrm,] )
    ## Right Limit ##
    rNrm = camRot.dot( R_y( np.radians( env_var("_D405_FOV_H_DEG")/2.0 ) ).dot( [1.0,0.0,0.0] ) )
    rPnt = cFocus.copy()
    rtnFOV.append( [rPnt, rNrm,] )
    ## Left Limit ##
    lNrm = camRot.dot( R_y( -np.radians( env_var("_D405_FOV_H_DEG")/2.0 ) ).dot( [-1.0,0.0,0.0] ) )
    lPnt = cFocus.copy()
    rtnFOV.append( [lPnt, lNrm,] )
    ## Return Limits ##
    return rtnFOV

########## BELIEFS #################################################################################


class ObjectMemory:
    """ Attempt to maintain recent and constistent object beliefs based on readings from the vision system """

    def reset_beliefs( self ):
        """ Remove all references to the beliefs, then erase the beliefs """
        self.beliefs : list[GraspObj] = []


    def __init__( self ):
        """ Set belief containers """
        self.reset_beliefs()
        
    
    def accum_evidence_for_belief( self, evidence : GraspObj, belief : GraspObj ):
        """ Use Bayesian multiclass update on `belief`, destructive """
        evdnc = extract_class_dist_in_order( evidence )
        prior = extract_class_dist_in_order( belief   )
        keys  = env_var("_BLOCK_NAMES")
        pstrr = multiclass_Bayesian_belief_update( 
            get_confusion_matx( env_var("_N_CLASSES"), confuseProb = env_var("_CONFUSE_PROB") ), 
            prior, 
            evdnc 
        )
        for i, key in enumerate( keys ):
            belief.labels[ key ] = pstrr[i]


    ##### Sensor Placement ################################################


    def p_symbol_in_cam_view( self, camXform : np.ndarray, symbol : GraspObj ):
        bounds = get_D405_FOV_frustum( camXform )
        qPosn  = extract_pose_as_homog( symbol )[0:3,3]
        blcRad = np.sqrt( 3.0 * (env_var("_BLOCK_SCALE")/2.0)**2 )
        return p_sphere_inside_plane_list( qPosn, blcRad, bounds )


    def integrate_one_reading( self, objReading : GraspObj, camXform : np.ndarray = None, 
                               maxRadius = 3.0*env_var("_BLOCK_SCALE"), suppressNew = False ):
        """ Fuse this belief with the current beliefs """
        relevant = False
        tsNow    = now()

        # 1. Determine if this belief provides evidence for an existing belief
        dMin     = 1e6
        belBest  = None
        for belief in self.beliefs:
            d = euclidean_distance_between_symbols( objReading, belief )

            # if not self.p_symbol_in_cam_view( camXform, belief ):
            #     print( f"\t\t{belief} not in cam view!, Distance: {d}" )

            if (d <= maxRadius) and (d < dMin) and ((camXform is None) or self.p_symbol_in_cam_view( camXform, belief )):
                dMin     = d
                belBest  = belief
                relevant = True

        if relevant:
            belBest.visited = True
            self.accum_evidence_for_belief( objReading, belBest )

            # updtFrac = objReading.score / (belBest.score + objReading.score)
            updtFrac = 0.85
            
            ## Update Pose ##
            belPosn = posn_from_xform( extract_pose_as_homog( belBest.pose    ) )
            objPosn = posn_from_xform( extract_pose_as_homog( objReading.pose ) )
            updPosn = objPosn * updtFrac + belPosn * (1.0 - updtFrac)
            updPose = np.eye(4)
            updPose[0:3,3] = updPosn
            belBest.pose  = ObjPose( updPose )

            ## Update Score ##
            belBest.count += objReading.count
            set_quality_score( belBest )
            belBest.ts = tsNow

        # 2. If this evidence does not support an existing belief, it is a new belief
        elif p_symbol_inside_workspace_bounds( objReading ) and (not suppressNew):
            print( f"\tNO match for {objReading}, Append to beliefs!" )
            nuBel = objReading.copy()
            nuBel.LKG = False
            self.beliefs.append( nuBel ) 

        # N. Return whether the reading was relevant to an existing belief
        return relevant
    

    def integrate_null( self, belief : GraspObj, avgScore = None ):
        """ Accrue a non-observation """
        labels = get_confused_class_reading( env_var("_NULL_NAME"), env_var("_CONFUSE_PROB"), env_var("_BLOCK_NAMES") )
        cnfMtx = get_confusion_matx( env_var("_N_CLASSES"), env_var("_CONFUSE_PROB") )
        priorB = [ belief.labels[ label ] for label in env_var("_BLOCK_NAMES") ] 
        evidnc = [ labels[ label ] for label in env_var("_BLOCK_NAMES") ]
        updatB = multiclass_Bayesian_belief_update( cnfMtx, priorB, evidnc )
        belief.labels = {}
        for i, name in enumerate( env_var("_BLOCK_NAMES") ):
            belief.labels[ name ] = updatB[i]
        if avgScore is not None:
            belief.score = exp_filter( belief.score, avgScore, env_var("_SCORE_FILTER_EXP") )
        # 2024-07-26: NOT updating the timestamp as NULL evidence should tend to remove a reading from consideration
    

    def unvisit_beliefs( self ):
        """ Set visited flag to False for all beliefs """
        for belief in self.beliefs:
            belief.visited = False


    def erase_dead( self ):
        """ Erase all beliefs and cached symbols that no longer have relevancy """
        retain = []
        for belief in self.beliefs:
            if (belief.labels[ env_var("_NULL_NAME") ] < env_var("_NULL_THRESH")) and ((now() - belief.ts) <= env_var("_OBJ_TIMEOUT_S")):
                retain.append( belief )
            elif env_var("_VERBOSE"):
                print( f"{str(belief)} DESTROYED!" )
        self.beliefs = retain


    def decay_beliefs( self, camXform : np.ndarray ):
        """ Destroy beliefs that have accumulated too many negative indications """
        vstScores = list()
        for belief in self.beliefs:
            if belief.visited:
                vstScores.append( belief.score )
        if len( vstScores ):
            nuScore = np.mean( vstScores )
        else:
            nuScore = env_var("_DEF_NULL_SCORE")
        for belief in self.beliefs:
            if (not belief.visited) and self.p_symbol_in_cam_view( camXform, belief ):
                self.integrate_null( belief, avgScore = nuScore )
        self.erase_dead()
        self.unvisit_beliefs()


    def belief_update( self, evdncLst : list[GraspObj], camXform : np.ndarray, maxRadius = 3.0*env_var("_BLOCK_SCALE") ):
        """ Gather and aggregate evidence """

        ## Integrate Beliefs ##
        cNu = 0
        cIn = 0
        self.unvisit_beliefs()
        
        if not len( self.beliefs ):
            # WARNING: ASSUMING EACH OBJECT IS REPRESENTED BY EXACTLY 1 READING
            for objEv in evdncLst:
                if p_symbol_inside_workspace_bounds( objEv ):
                    self.beliefs.append( objEv )
        else:
            for objEv in evdncLst:
                # if not objEv.visitRD:
                #     objEv.visitRD = True
                if self.integrate_one_reading( objEv, camXform, maxRadius = maxRadius ):
                    cIn += 1
                else:
                    cNu += 1
            ## Decay Irrelevant Beliefs ##
            if env_var("_NULL_EVIDENCE"):
                self.decay_beliefs( camXform )

        if env_var("_VERBOSE"):
            if (cNu or cIn):
                print( f"\t{cNu} new object beliefs this iteration!" )
                print( f"\t{cIn} object beliefs updated!" )
            else:
                print( f"\tNO belief update!" )
        
        if env_var("_VERBOSE"):
            print( f"Total Beliefs: {len(self.beliefs)}" )
            for bel in self.beliefs:
                print( f"\t{bel}" )
            print()


    
