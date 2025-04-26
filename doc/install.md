## 安装环境

```
apt install ninja-build  # enable parallel compilation
apt install libomp-dev   # enable computational parallel optimization (OpenMP)
apt install libgl1-mesa-glx libglib2.0-0 # fix OpenCV missing lib

*using python=3.10 ubuntu2204 cuda12.1 docker sm_80

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/dominikandreas/nuscenes-devkit.git@feature/python312#subdirectory=setup -v
pip install -r requirements.txt -v

pip install -e . -v --no-build-isolation --force-reinstall
```

## 准备数据

BEVdet

升级到最新格式

```
python tools/dataset_converters/update_infos_to_v2.py --dataset nuscenes --pkl-path data/nuscenes/bevdetv2-nuscenes_infos_val.pkl --out-dir ./
python tools/dataset_converters/update_infos_to_v2.py --dataset nuscenes --pkl-path data/nuscenes/bevdetv2-nuscenes_infos_train.pkl --out-dir ./
```

然后手动替换掉原始pkl文件
