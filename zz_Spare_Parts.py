import numpy as np

    def closest_symbol_to_pose( self, pose, margin = None ) -> ThinSymbol:
        """ Fetch the closest symbol to the pose within `margin`, otherwise return None """
        if margin is None:
            margin = 2.0 * env_var("_BLOCK_SCALE")
        pose = extract_pose_as_homog( pose )
        dMin = 1e9
        sMin = None
        for v in self.symH.values():
            d = translation_diff( pose, v.pose )
            if d < dMin:
                dMin = d
                sMin = v
        if dMin <= margin:
            return sMin
        else:
            return None


    def move_symbol_from_to_pose( self, srcPose, dstPose ):
        """ Find the symbol at `srcPose` and move it to `dstPose`, Return thin symbols if it was moved, else return None """
        dstPose  = extract_pose_as_homog( dstPose )
        self.bMem.update_belief_pose( extract_pose_as_homog( srcPose ), dstPose )
        needMove = self.closest_symbol_to_pose( srcPose )
        if (needMove is not None):
            needMove.pose = dstPose.copy()
            return needMove
        else:
            return None
        

    def fail_symbol( self, srcPose ):
        """ Stop believing in the thing we tried to move """
        needFail = self.closest_symbol_to_pose( srcPose )
        del self.symH[ needFail.id ]
        self.bMem.del_beliefs_close_to_pose( srcPose )


    def update_symbol_history( self, symLst : list[GraspObj] ):
        """ Match new symbols to current and calculate confidence changes """
        for sym in symLst:
            tSm = self.closest_symbol_to_pose( sym )
            if tSm is None:
                nuS = ThinSymbol( label = sym.label, pose = sym.pose )
                nuS.append_dist( sym.labels )
                self.symH[ nuS.id ] = nuS
            else:
                tSm.append_dist( sym.labels )


    def check_KL_for_symbol_at_pose( self, pose, expectedLabel : str, poseMargin : float = None, N_falling : int = 3 ):
        """ Return `check_KL_criteria` for the symbol nearest this pose """
        chkSym = self.closest_symbol_to_pose( pose, margin = poseMargin )
        print( f"About to check {expectedLabel} @ {pose}, found {chkSym}" )
        if chkSym is not None:
            res = chkSym.check_KL_criteria( N_falling, expectedLabel )
            print( f"Result?: {res}" )
            return 
        else:
            return False
    
def p_bb_intersect( boxA, boxB ):
    """ Return true if the 2D bounding boxes intersect """
    # Original Author: Dennis Bauszus, https://stackoverflow.com/a/77133433
    return (not ((boxA[0] > boxB[2]) or (boxA[2] < boxB[0]) or (boxA[1] > boxB[3]) or (boxA[3] < boxB[1])))


def pos_mask_from_bbox( shape, bbox ):
    """ Return an array of `shape` where all entries inside the 2D `bbox` are 1, and everything else is 0 """
    bbox   = [int(c) for c in bbox]
    rntMtx = np.zeros( shape[:2] )
    rntMtx[bbox[1]:bbox[3], bbox[0]:bbox[2],] = np.ones( (bbox[3]-bbox[1], bbox[2]-bbox[0],) )
    return rntMtx


def mask_subtract( mask1, mask2 ):
    """ Subtract `mask2` from `mask1` """
    return np.clip( np.subtract( mask1, mask2 ), 0, 1 )
    

def avg_color_in_mask( image, mask ):
    """ Return the average `image` color where `mask` is True """
    if np.sum( mask ) < 1.0:
        return np.zeros( (3,) )
    nuMsk = np.zeros( image.shape ) 
    for i in range(3):
        nuMsk[:,:,i] = mask
    return np.mean( image, axis = (0,1), where = nuMsk > 0.005 )
    

def p_bbox_contains_other( boxA, boxB ):
    """ Return [ <A contains B>, <B contains A> ] """
    return [
        (boxA[0] <= boxB[0]) and (boxA[1] <= boxB[1]) and (boxA[2] >= boxB[2]) and (boxA[3] >= boxB[3]),
        (boxB[0] <= boxA[0]) and (boxB[1] <= boxA[1]) and (boxB[2] >= boxA[2]) and (boxB[3] >= boxA[3]),
    ]

def mask_ray( mask : np.ndarray, bbox : np.ndarray ):
    """ Project a ray through the center of the mask """
    rows   = mask.shape[0]
    rwHf   = rows / 2
    cols   = mask.shape[1]
    clHf   = cols / 2
    cntr2d = np.zeros( 2 )
    count  = 0.0
    Xlen   = np.tan( np.radians( env_var("_D405_FOV_H_DEG")/2.0 ) ) 
    Ylen   = np.tan( np.radians( env_var("_D405_FOV_V_DEG")/2.0 ) ) 
    for j in range( bbox[1], min(bbox[3]-1, rows) ):
        for k in range( bbox[0], min(bbox[2]-1, cols) ):
            # print( j,k )
            frac_jk =  mask[j,k]
            cntr2d  += np.array( [(k-clHf)/clHf,(j-rwHf)/rwHf] ) * frac_jk
            count   += frac_jk
    if count > 0.0:
        cntr2d /= count
    return vec_unit( [cntr2d[0]*Xlen, cntr2d[1]*Ylen, 1.0] )