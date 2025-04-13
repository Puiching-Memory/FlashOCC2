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

### OpenOccupancy

官方数据集文档:
[https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md)

下载:预生成pkl
[https://github.com/JeffWang987/OpenOccupancy/releases/tag/train_pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/train_pkl)

[https://github.com/JeffWang987/OpenOccupancy/releases/tag/val_pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/val_pkl)

下载:OCC数据

| name          | 谷歌云盘                                                                                  | 百度云                                               | 大小  |
| ------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----- |
| trainval-v0.1 | [Google](https://drive.google.com/file/d/1vTbgddMzUN6nLyWSsCZMb9KwihS7nPoH/view?usp=sharing) | [25UE](https://pan.baidu.com/s/1Wu1EYa7vrh8KS8VPTIny5Q) | ~=5GB |

```
flashocc2/
├── data/
    ├── depth_gt/ (generate by create_data_OpenOccupancy.py) (204894) (26G)
    ├── nuScenes-Occupancy/ (download OCC) (34149) (135G)
    ├── nuscenes/
    |    ├── maps/
    |    ├── samples/
    |    ├── sweeps/
    |    ├── lidarseg/
    |    ├── v1.0-test/
    |    ├── v1.0-trainval/
    |    ├── nuscenes_occ_infos_train.pkl/
    |    ├── nuscenes_occ_infos_val.pkl/
  
```

预生成深度图:

```
python ./tools/create_data_OpenOccupancy.py
```
