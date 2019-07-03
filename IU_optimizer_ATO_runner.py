# This file loads the optimizer, loads the test problem, then it
# applies the optimizer to the test problem and saves the outputs.
# This is just rough draft code and almost definitely doesn't work!
# I have just put it here to show how to structure the code :)


from IU_optimizer.Input_Uncertainty import Multi_Input_Uncert
from IU_optimizer.utils import toysource as atosource
from TestProblems.ATO import assemble_to_order as atofun
import np as np
import pandas as pd
from pprint import pprint

# initilize the optimizer
myoptimizer = Multi_Input_Uncert(f=atofun,
                                 xran=atofun.xran,
                                 wran=atofun.wran,
                                 other_shit_required_to_by_delta_loss....)

# now run the optimizer 100 times and save all outputs
OC= []
for i in range(100):
    identifier = 100 + np.random.random()

    OC ,N_I, var = myoptimizer(init_sample=10, iu_init=10, EndN=100, seed=i)
    
    data = {'OC': OC,
            'len': [N_I]*len(OC),
            'var1':[var[0]]*len(OC),
            'var2':[var[1]]*len(OC)}

    pprint('data',data)

    gen_file = pd.DataFrame.from_dict(data)
    path ='/home/rawsys/matjiu/PythonCodes/PHD/Input_Uncertainty/With_Input_Selection/Data_MC100/OC_'
    path = path +str(i)+'_' + str(identifier) + '.csv'
    gen_file.to_csv(path_or_buf=path)
