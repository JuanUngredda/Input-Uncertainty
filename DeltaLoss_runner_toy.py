from IU_optimizer import DeltaLoss
from IU_optimizer.utils import toyfun
from IU_optimizer.utils import toysource


output = DeltaLoss(simulator = toyfun,
                   xran = toyfun.x_range,
                   wran = toyfun.w_ran,
                   external_source = toysource,
                   iu_distro="Gaussian"
                   )
