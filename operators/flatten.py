import torch.nn as nn
from intervals.symbolic_interval import SymbolicInterval
import torch
from copy import deepcopy



class Flatten(nn.Module):
    def __init__(self, torch_layer):
        super().__init__()
        self.torch_layer = torch_layer
        self.pre_symbolic = None
        self.post_symbolic = None
        

    def forward(self, x : SymbolicInterval):
        """
        Parameters
        ----------
        x: Symbolic Interval Object
        """
        self.pre_symbolic = x
        post_interval = SymbolicInterval(x.input_interval, x.l.clone(), x.u.clone())
        post_interval.concretize()
        self.post_symbolic = post_interval
        return post_interval

    @property
    def pre_conc_bounds(self):
        return self.pre_symbolic.conc_bounds

    @property 
    def post_conc_bounds(self):
        return self.post_symbolic.conc_bounds




        

