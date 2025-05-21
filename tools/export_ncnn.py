import sys
import os

sys.path.append(os.path.abspath("./"))

from lib.utils.logger import logger
from lib.cfg.base import DataSetBase

import torch
import importlib
import argparse
from torchinfo import summary
import pnnx

def export(model, device, test_loader, logger):
    # os.makedirs(os.path.join(args.cfg, "checkpoint", "ncnn"),exist_ok=True)
    logger.info(f"Forward once to generate ncnn model ...")
    for inputs, targets, data_info in test_loader:
        inputs = inputs.to(device)

        opt_model = pnnx.export(
            model, os.path.join(args.cfg, "checkpoint", "model.ncnn"), inputs
        )
        break

    logger.info(
        f"Successfully exported ncnn model to {os.path.join(args.cfg, "checkpoint")}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite export script")
    parser.add_argument("--cfg", dest="cfg", help="path to config file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 添加模块搜索路径
    sys.path.append(args.cfg)

    # 导入模型
    model: torch.nn.Module = importlib.import_module("model").model()
    model.eval()

    checkpoint_dict = torch.load(
        os.path.join(args.cfg, "checkpoint", "model.pth"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint_dict["model"])
    model = model.to(device)

    # 导入数据集
    data_set: DataSetBase = importlib.import_module("dataset").data_set()

    # 打印基本信息
    print(
        f"\n{summary(model, input_size=(data_set.get_bath_size(),3,384,1280),mode='train',verbose=0)}"
    )

    export(
        model,
        device,
        data_set.get_test_loader(),
        logger,
    )
