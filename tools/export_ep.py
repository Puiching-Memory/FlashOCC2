import sys
import os

sys.path.append(os.path.abspath("./"))

from lib.utils.logger import logger
from lib.cfg.base import DataSetBase

import torch
import importlib
import argparse
from torchinfo import summary
import torch_tensorrt


def export(model, device, test_loader, logger):
    logger.info(f"Forward once to generate TensorRT model (exported_program) ...")
    for inputs, targets, data_info in test_loader:
        inputs = [inputs.to("cuda")] # FIXME: half() causes error

        enabled_precisions = {torch.float}
        trt_ts_module = torch_tensorrt.compile(
            model,ir="dynamo", inputs=inputs, enabled_precisions=enabled_precisions
        )
        torch_tensorrt.save(trt_ts_module, os.path.join(args.cfg, "checkpoint", "model.ep"),output_format="exported_program",inputs=inputs)
        logger.info(
            f"Successfully exported TensorRT model (exported_program) to {os.path.join(args.cfg, "checkpoint", "model.ep")}"
        )
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite export script")
    parser.add_argument("--cfg", dest="cfg", help="path to config file")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")

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
