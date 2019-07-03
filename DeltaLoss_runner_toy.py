from IU_optimizer.input_uncertainty import DeltaLoss
from IU_optimizer.utils import toyfun
from IU_optimizer.utils import toysource


output = DeltaLoss(test_fun = toyfun,
                   xran = toyfun.x_range,
                   wran = toyfun.w_ran,
                   info_source = toysource,
                   iu_distro="Gaussian"
                   )
