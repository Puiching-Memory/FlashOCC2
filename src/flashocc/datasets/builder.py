"""数据集注册表 — 保留 Registry 仅供 @register_module 装饰器内部使用."""
from flashocc.core import Registry

DATASETS = Registry('datasets')
PIPELINES = Registry('pipelines')
