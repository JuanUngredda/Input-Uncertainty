# If you import things from the files into this file, then
# they are exposed from outside the dir
# eg in the runner_(problem).py files, instead of 
#  'from TestProblems.Toy import toysource' or
#  'from TestProblems.ato_hongnelson import ATO'
#
# you can use   
#   'from TestProblems import toysource'
#   'from TestProblems import ATO'
#
# this is really useful when there are lots of dirs and subdirs
# because you can  avoid `from dir1.dir2.dir3.dir4 import myfun`

from .Toy import toysource, toyfun, GP_test
from .ato_hongnelson import ATO_HongNelson
from .Ambulance_simulator import Ambulance_Delays
from .experiments2d import rosenbrock
from .newsvendor import newsvendor_deterministic
from .newsvendor import newsvendor_noisy
