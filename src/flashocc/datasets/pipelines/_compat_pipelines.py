"""mmdet3d.datasets.pipelines — data processing pipeline components."""
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadPointsFromFile:
    """Load 3D points from file.

    Args:
        coord_type (str): Coordinate type ('LIDAR', 'CAMERA', 'DEPTH').
        load_dim (int): Dimensions of the loaded points.
        use_dim (list[int] | int): Dimensions to use.
    """

    def __init__(self, coord_type='LIDAR', load_dim=5, use_dim=None,
                 shift_height=False, use_color=False, **kwargs):
        self.coord_type = coord_type
        self.load_dim = load_dim
        if use_dim is None:
            self.use_dim = list(range(load_dim))
        elif hasattr(use_dim, '__index__'):
            # int-like: has __index__ method
            self.use_dim = list(range(use_dim))
        else:
            self.use_dim = list(use_dim)
        self.shift_height = shift_height
        self.use_color = use_color

    def __call__(self, results):
        pts_filename = results.get('pts_filename', '')
        points = self._load_points(pts_filename)
        points = points[:, self.use_dim]
        from flashocc.core.bbox.points import get_points_type
        coord_map = {'LIDAR': 0, 'CAMERA': 1, 'DEPTH': 2}
        mode = coord_map.get(self.coord_type, 0)
        points_class = get_points_type(mode)
        pts_obj = points_class(points, points_dim=points.shape[-1])
        results['points'] = pts_obj
        return results

    @staticmethod
    def _load_points(filename):
        """Load points from .bin or .npy file."""
        if filename.endswith('.npy'):
            return np.load(filename).astype(np.float32)
        return np.fromfile(filename, dtype=np.float32).reshape(-1, 5)

    def __repr__(self):
        return (f'{self.__class__.__name__}(coord_type={self.coord_type}, '
                f'load_dim={self.load_dim}, use_dim={self.use_dim})')


@PIPELINES.register_module()
class MultiScaleFlipAug3D:
    """Test-time augmentation wrapper (simplified — no multi-scale / no flip).

    When ``flip=False`` and only one scale is provided, this simply applies the
    inner *transforms* sequentially, which is the most common inference setup.
    """

    def __init__(self, transforms=None, img_scale=None,
                 pts_scale_ratio=1, flip=False, pcd_horizontal_flip=False,
                 pcd_vertical_flip=False, **kwargs):
        from flashocc.datasets.pipelines.compose import Compose
        self.transforms = Compose(transforms or [])
        self.img_scale = img_scale
        self.pts_scale_ratio = pts_scale_ratio
        self.flip = flip

    def __call__(self, results):
        return self.transforms(results)

    def __repr__(self):
        return (f'{self.__class__.__name__}(flip={self.flip}, '
                f'img_scale={self.img_scale})')


__all__ = [
    'LoadPointsFromFile', 'MultiScaleFlipAug3D',
]
