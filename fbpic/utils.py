"""
Generic utility functions for the package hankel_dt
"""

def array_multiply( a, v, axis ) :
    """
    Mutliply the array `a` (of any shape) with the vector
    `v` along the axis `axis`
    The axis `axis` of `a` should have the same length as `v`

    Parameters :
    ------------
    a : ndarray (real or complex values)
    The array to be multiplied by v

    v : 1darray
    The 1d vector used for the multiplication

    axis : int
    The axis of `a` along which the multiplication is carried out

    Returns :
    --------
    A matrix of the same shape as a
    """
    
    if a.ndim > 1 and axis!=-1 :
        # Carry out the product by moving the axis `axis`
        # (This is done because the shape of the array a is
        # unknown, and thus the syntax for multiplying is unclear.
        # The axis -1 is chosen because it is the fastest index
        # for multiplying.)
        r = a.swapaxes(-1,axis)
    else :
        r = a
    # Calculate the product    
    r = r*v
    # If needed swap the axes back
    if a.ndim > 1 and axis!=-1 :
        r = r.swapaxes(-1,axis)

    return(r)
 
