import sys
import os

sys.path.append(os.path.abspath("./"))

from lib.utils.logger import get_logger
from lib.models.init import weight_init
from lib.cfg.base import (
    LossBase,
    TrainerBase,
    DataSetBase,
    SchedulerBase,
    OptimizerBase,
    VisualizerBase,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp
import numpy as np
import random

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import importlib
import argparse
from torchinfo import summary
import time
import swanlab
import datetime
from typing import Optional
from tqdm import tqdm

from swanlab.log import trace_handler

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = -1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pid = os.getpid()  # 获取进程id
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 设置同步cuda,仅debug时使用
# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"

def train(
    model: torch.nn.Module,
    ema_model: Optional[torch.nn.Module],
    trainner: TrainerBase,
    device,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: LossBase,
    visualizer: VisualizerBase,
    logger,
):
    scaler = torch.amp.GradScaler()

    # pin random seed
    logger.info(f"set random seed: {trainner.get_seed()}")
    torch.manual_seed(trainner.get_seed())
    np.random.seed(trainner.get_seed())
    random.seed(trainner.get_seed())

    # profiler
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/'),
    #     on_trace_ready=trace_handler(),
    #     with_flops=True,
    #     record_shapes = True,
    #     profile_memory = True,
    #     with_stack = True,
    # ) as p:

    # epoch循环
    microstep = 0
    bar_epoch = tqdm(
        total=trainner.get_end_epoch() - trainner.get_start_epoch(),
        position=0,
        leave=True,
    )
    for epoch in range(trainner.get_start_epoch(), trainner.get_end_epoch()):
        epoch_start_time = time.perf_counter_ns()
        model.train()
        # dataset循环
        bar_dataset = tqdm(train_loader, position=1, leave=False)
        for index, (inputs, targets, data_info) in enumerate(train_loader):
            optimizer.zero_grad()
            # inputs = inputs.to(device,memory_format=torch.channels_last)
            inputs = inputs.to(device)
            targets = {key: value.to(device) for key, value in targets.items()}
            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=trainner.is_amp(),
            ):
                forward_time = time.perf_counter_ns()
                outputs = model(inputs)
                forward_time = (time.perf_counter_ns() - forward_time) / 1e6  # ms

                loss_time = time.perf_counter_ns()
                loss, loss_info = loss_fn(outputs, targets)
                loss_time = (time.perf_counter_ns() - loss_time) / 1e6  # ms

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                ema_model.update_parameters(model)

            info_dataset = {
                "microstep": (microstep := microstep + 1),
                "forward_time(ms)": forward_time,
                "loss_time(ms)": loss_time,
                "dataload_time(ms)": round(
                    torch.mean(data_info["dataload_time"]).item(), 4
                ),
                "loss": loss,
                **loss_info,
            }
            swanlab.log(info_dataset)
            bar_dataset.set_postfix({"loss": round(loss.item(), 2)})
            bar_dataset.update()
            break

        scheduler.step()

        # 保存模型
        if not os.path.exists(trainner.get_save_path()):
            os.mkdir(trainner.get_save_path())
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            os.path.join(trainner.get_save_path(), "model.pth"),
        )
        logger.info(f"checkpoint: {epoch} saved to {trainner.get_save_path()}")

        # # 保存模型预测可视化结果
        # model.eval()
        # results = visualizer.decode_output(inputs, outputs, data_info)
        # results = {key: swanlab.Image(value) for key, value in results.items()}
        # swanlab.log(results)

        # # 保存真值可视化结果
        # results = visualizer.decode_target(inputs, targets, data_info)
        # results = {key: swanlab.Image(value) for key, value in results.items()}
        # swanlab.log(results)

        logger.info(
            f"epoch {epoch} finished, time: {time.perf_counter_ns()-epoch_start_time:.2f}s"
        )

        if ema_model is not None:
            logger.info("update bn with ema model, it may takes few minutes ...")
            torch.optim.swa_utils.update_bn(train_loader, ema_model, device=device)
            logger.info(
                f"ema checkpoint saved to {os.path.join(trainner.get_save_path(), "model_ema.pth")}"
            )
            torch.save(
                {
                    "model": ema_model.state_dict(),
                },
                os.path.join(trainner.get_save_path(), "model_ema.pth"),
            )

        info_epoch = {
            "epoch": epoch,
        }

        bar_epoch.set_postfix(info_epoch)
        bar_epoch.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite training script")
    parser.add_argument(
        "--cfg",
        dest="cfg",
        default=r"C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet",
        help="path to config file",
    )
    args = parser.parse_args()

    logger = get_logger()

    # 初始化swanlab,启动$swanlab watch ./logs
    try:
        swanlab.init(
            project="monolite",
            experiment_name=f"{os.path.basename(args.cfg)}_{datetime.datetime.now().strftime('%Y/%m/%d_%H:%M:%S')}",
        )
    except Exception as e:
        logger.error(f"{e}\nInit swanlab in local mode")
        swanlab.init(
            project="monolite",
            experiment_name=f"{os.path.basename(args.cfg)}_{datetime.datetime.now().strftime('%Y/%m/%d_%H:%M:%S')}",
            logdir="./logs",  # 本地模式
            mode="local",
        )
    # 添加模块搜索路径
    sys.path.append(args.cfg)
    logger.info(f"searching module in {args.cfg}")

    # 导入训练配置
    trainner: TrainerBase = importlib.import_module("trainner").trainner()

    # 导入模型
    model: torch.nn.Module = importlib.import_module("model").model()

    # 导入数据集
    data_set: DataSetBase = importlib.import_module("dataset").data_set()

    # 打印模型信息
    print(
        f"\n{summary(model, input_size=(data_set.get_bath_size(),3,384,1280),mode='train',verbose=0,depth=2)}"
    )

    # 导入优化器
    optimizer: OptimizerBase = importlib.import_module("optimizer").optimizer(model)
    optimizer: torch.optim.Optimizer = optimizer.get_optimizer()

    # 导入学习率衰减器
    scheduler: SchedulerBase = importlib.import_module("scheduler").scheduler(optimizer)
    scheduler: torch.optim.lr_scheduler.LRScheduler = scheduler.get_scheduler()

    # 导入损失函数
    loss: LossBase = importlib.import_module("loss").loss()

    # 导入可视化工具
    visualizer: VisualizerBase = importlib.import_module("visualizer").visualizer()

    # 从checkpoint恢复训练
    if trainner.get_resume_checkpoint() == None:
        model.apply(weight_init)  # 权重初始化
    else:
        logger.info(f"resume from {trainner.get_resume_checkpoint()}")
        checkpoint_dict = torch.load(
            trainner.get_resume_checkpoint(),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint_dict["model"])
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        scheduler.load_state_dict(checkpoint_dict["scheduler"])
        trainner.set_start_epoch(checkpoint_dict["epoch"])

    # 尝试编译模型及函数
    # try:
    #     logger.warning(f"try to compile model...")
    #     model = torch.compile(model)
    #     logger.warning(f"try to compile functions...")
    #     loss = torch.compile(loss)
    # except Exception as e:
    #     logger.error(f"compile model error: {e}")

    # model = model.to(device,memory_format=torch.channels_last)
    model = model.to(device)

    # 构建EMA模型
    # ema_model = torch.optim.swa_utils.AveragedModel(
    #     model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
    # )

    # 打印训练配置信息
    logger.info(data_set)
    logger.info(optimizer)
    logger.info(scheduler)

    # 开始训练
    train(
        model,
        None,
        trainner,
        device,
        data_set.get_train_loader(),
        data_set.get_test_loader(),
        optimizer,
        scheduler,
        loss,
        visualizer,
        logger,
    )
