import numpy as np


def toyfun(x, w):
    #
    # ARGS
    #  x: scalar decision variable
    #  w: scalaer input parameter
    #
    # RETURNS
    #  y: scalar output function

    assert len(x)==1, "toyfun only works with 1D x!"
    assert len(w)==1, "toyfun only works with 1D w!"

    r = np.sqrt( ((x-0.2)**2 + (w-0.2)**2) )

    y =  np.cos(r*2.*np.pi) / (r+2)

    return(y)

# set valid input ranges as function attributes
setattr(toyfun, 'x_ran', np.array([0.,1.]))
setattr(toyfun, 'w_ran', np.array([0.,1.]))


def toysource(n_samples=1):
    # a Gaussain centrered at 0.5, the middle of w_range for the toyfun
    return( 0.5 + np.random.normal(size = (n_samples)) )
