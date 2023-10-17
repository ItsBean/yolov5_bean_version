import torch.nn as nn
from models.common import *  # noqa
from models.experimental import *  # noqa
m = 'Conv'
m = eval(m) if isinstance(m, str) else m  # eval strings

# Test
print(m)  # This should print something like <class 'torch.nn.modules.conv.Conv2d'>
