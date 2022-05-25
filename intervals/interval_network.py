'''
Class that converts torch model to interval network. 
Propagating an interval through this network computes bounds on all neurons using an underlying Interval bound propagation method (SIA,IA,..) 
'''
import torch.nn as nn
import torch
from utils.Logger import get_logger
logger = get_logger(__name__)

class IntervalNetwork(nn.Module):
    def __init__(self, model: nn.Module, operators_dict: dict):
        super().__init__()
        # self._model = model
        self.operators_dict = operators_dict
        self.layers = []

        for module in list(model.modules())[1:]:
            module_name = str(module).split('(')[0]
            if 'Sequential' in module_name:
                continue
            try:
                self.layers.append(self.operators_dict[module_name](module))
            except Exception as e:
                logger.exception(f"Operation {module_name} not implemented")
                raise 

       
        self.interval_net = nn.Sequential(*self.layers)

    
    def forward(self, interval):
        return self.interval_net(interval)




