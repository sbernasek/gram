import sys
from os.path import abspath
p = abspath('/Users/Sebi/Documents/grad_school/research/genessa/genessa')
if p not in sys.path:
    sys.path.insert(0, p)

from matplotlib import rcParams, font_manager
rcParams['font.family'] = "Helvetica"
#rcParams['font.sans-serif'] = ['Arial']
font_manager._rebuild()
