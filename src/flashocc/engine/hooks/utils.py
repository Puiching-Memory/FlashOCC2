# Copyright (c) OpenMMLab. All rights reserved.

__all__ = ['is_parallel']


def is_parallel(model):
    """check if model is in parallel mode."""
    return hasattr(model, 'module') and hasattr(model, 'device_ids')
