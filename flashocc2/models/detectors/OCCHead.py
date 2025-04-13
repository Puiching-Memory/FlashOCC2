from typing import Optional, Union, Dict

import torch

from mmengine.model import BaseModel
from mmdet3d.registry import MODELS


@MODELS.register_module()
class OCCHead(BaseModel):

    def __init__(self):
        pass

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:

        return