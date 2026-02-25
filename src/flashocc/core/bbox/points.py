"""3D 点云数据结构."""
import torch
import numpy as np


class BasePoints:
    """3D 点基类."""
    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).float()
        self.tensor = tensor
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims or {}

    @property
    def coord(self):
        return self.tensor[:, :3]

    def __len__(self):
        return self.tensor.shape[0]

    def to(self, device):
        self.tensor = self.tensor.to(device)
        return self

    def numpy(self):
        return self.tensor.cpu().numpy()


class LiDARPoints(BasePoints):
    pass


_POINT_TYPES = {
    "LIDAR": LiDARPoints,
    "DEPTH": BasePoints,
}


def get_points_type(points_type):
    return _POINT_TYPES.get(points_type, BasePoints)


__all__ = ["BasePoints", "LiDARPoints", "get_points_type"]
