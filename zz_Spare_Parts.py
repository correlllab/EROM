
    
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