import sys
import os

sys.path.append(os.path.abspath("./"))

from lib.utils.logger import logger
from lib.cfg.base import DataSetBase, VisualizerBase

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = -1


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import importlib
import argparse
from torchinfo import summary
import time
from rich.progress import track


def detect(
    model,
    device,
    test_loader: torch.utils.data.DataLoader,
    visualizer: VisualizerBase,
    logger,
):

    for i, (inputs, targets, data_info) in track(enumerate(test_loader), "Detecting"):
        inputs = inputs.to(device)
        targets = {key: value.to(device) for key, value in targets.items()}

        forward_time = time.time_ns()
        outputs = model(inputs)
        forward_time = (time.time_ns() - forward_time) / 1e6  # ms

        if not os.path.exists(visualizer.save_path):
            os.mkdir(visualizer.save_path)

        result = visualizer.decode_output(inputs, outputs)
        result = visualizer.decode_target(inputs, targets)

        info = {
            "forward_time": forward_time,
        }

        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite detection script")
    parser.add_argument("--cfg", dest="cfg", help="path to config file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 添加模块搜索路径
    sys.path.append(args.cfg)

    # 导入模型
    model: torch.nn.Module = importlib.import_module("model").model()
    checkpoint_dict = torch.load(
        os.path.join(args.cfg, "checkpoint", "model.pth"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint_dict['model'])
    model.eval()
    model = model.to(device)

    # 导入数据集
    data_set: DataSetBase = importlib.import_module("dataset").data_set()

    # 导入可视化工具
    visualizer: VisualizerBase = importlib.import_module("visualizer").visualizer()

    # 打印基本信息
    print(
        f"\n{summary(model, input_size=(data_set.get_bath_size(),3,384,1280),mode='train',verbose=0)}"
    )
    logger.info(data_set)

    detect(
        model,
        device,
        data_set.get_test_loader(),
        visualizer,
        logger,
    )
