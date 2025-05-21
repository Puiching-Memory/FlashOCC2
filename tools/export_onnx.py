import sys
import os

sys.path.append(os.path.abspath("./"))

from lib.utils.logger import logger
from lib.cfg.base import DataSetBase

import torch
import importlib
import argparse
from torchinfo import summary
from onnxsim import simplify
import onnx


def export(model, device,test_loader, logger):
    logger.info(f"Forward once to generate onnx model ...")
    for inputs, targets, data_info in test_loader:
        inputs = inputs.to(device)
        #output = model(inputs)
        with open(os.path.join(args.cfg, "checkpoint", "model.onnx"), "wb") as f:
            onnx_program = torch.onnx.export(
                model,
                (inputs,),
                f,
                verbose=False,
                dynamo=False,
                opset_version=20,
                #input_names=["input"],
                #output_names=[""],
            )
            logger.info(f"Successfully exported ONNX model to {os.path.join(args.cfg, "checkpoint", "model.onnx")}")
        break
    
    logger.info("Simplifying ONNX model ...")
    # simplify onnx model
    model = onnx.load(os.path.join(args.cfg, "checkpoint", "model.onnx"))
    onnx.checker.check_model(model)
    # TODO:not support opset > 20 yet
    model_simp, check = simplify(model) 
    onnx.save(model_simp, os.path.join(args.cfg, "checkpoint", "model_simp.onnx"))
    logger.info(f"Successfully exported ONNX model to {os.path.join(args.cfg, "checkpoint", "model_simp.onnx")}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite export script")
    parser.add_argument("--cfg", dest="cfg", help="path to config file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 添加模块搜索路径
    sys.path.append(args.cfg)

    # 导入模型
    model:torch.nn.Module = importlib.import_module("model").model()
    model.eval()
    
    checkpoint_dict = torch.load(
        os.path.join(args.cfg, "checkpoint", "model.pth"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint_dict["model"])
    model = model.to(device)

    
    # 导入数据集
    data_set:DataSetBase = importlib.import_module("dataset").data_set()

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
