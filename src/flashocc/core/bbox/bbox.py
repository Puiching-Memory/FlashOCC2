"""3D 边界框数据结构."""
import torch
import numpy as np


class BaseInstance3DBoxes:
    """3D 边界框基类."""
    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).float()
        elif isinstance(tensor, (list, tuple)):
            tensor = torch.tensor(tensor, dtype=torch.float32)
        if tensor.numel() == 0:
            tensor = tensor.reshape(0, box_dim)
        self.tensor = tensor
        self.box_dim = box_dim
        self.with_yaw = with_yaw

    @property
    def bev(self):
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @property
    def gravity_center(self):
        return self.tensor[:, :3]

    @property
    def dims(self):
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        return self.tensor[:, 6]

    def __len__(self):
        return self.tensor.shape[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(n={len(self)})"

    def to(self, device):
        self.tensor = self.tensor.to(device)
        return self

    def numpy(self):
        return self.tensor.cpu().numpy()

    @property
    def device(self):
        return self.tensor.device


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """LiDAR 坐标系 3D 框."""
    pass


__all__ = ["BaseInstance3DBoxes", "LiDARInstance3DBoxes"]
