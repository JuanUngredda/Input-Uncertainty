# This file loads the optimizer, loads the test problem, then it
# applies the optimizer to the test problem and saves the outputs.
# This is just rough draft code and almost definitely doesn't work!
# I have just put it here to show how to structure the code :)


from IU_optimizer.Input_Uncertainty import Multi_Input_Uncert
from IU_optimizer.utils import toyfun, toysource
import np as np
import pandas as pd
from pprint import pprint

# initilize the optimizer
myoptimizer = Multi_Input_Uncert(f=toyfun,
                                 xran=toyfun.xran,
                                 wran=toyfun.wran,
                                 other_shit_required_to_by_delta_loss....)

# now run the optimizer 100 times and save all outputs

OC ,N_I, var = myoptimizer(init_sample=10, iu_init=10, EndN=100, seed=i)

data = {'OC': OC,
        'len': [N_I]*len(OC),
        'var1':[var[0]]*len(OC),
        'var2':[var[1]]*len(OC)}

pprint('data',data)
t