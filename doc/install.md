## Environment Setup

step 1. Install environment for pytorch training

```
install uv env manager,see: https://docs.astral.sh/uv/getting-started/first-steps/

uv python install 3.12

apt install ninja-build  # enable parallel compilation
apt install libomp-dev   # enable computational parallel optimization (OpenMP)
apt install libgl1-mesa-glx libglib2.0-0 # fix OpenCV missing libGL.so.1

uv sync --extra {cpu,cu121} --no-build-isolation
```

step 2. Prepare nuScenes dataset as introduced in [nuscenes_det.md](nuscenes_det.md) and create the pkl for FlashOCC by running:

```shell
uv run tools/create_data_bevdet.py
```

thus, the folder will be ranged as following:

```shell
└── Path_to_FlashOcc2/
    └── data
        └── nuscenes
            ├── v1.0-trainval (existing)
            ├── sweeps  (existing)
            ├── samples (existing)
	    ├── lidarseg(existing)
            ├── maps    (existing)
            ├── bevdetv2-nuscenes_infos_train.pkl (new)
            └── bevdetv2-nuscenes_infos_val.pkl (new)
```

step 3. For Occupancy Prediction task, download (only) the 'gts' from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and arrange the folder as:

```shell
└── Path_to_FlashOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval (existing)
            ├── sweeps  (existing)
            ├── samples (existing)
            ├── gts (new)
            ├── bevdetv2-nuscenes_infos_train.pkl (new)
            └── bevdetv2-nuscenes_infos_val.pkl (new)
```

(for panoptic occupancy), we follow the data setting in SparseOcc:

(1) Download Occ3D-nuScenes occupancy GT from [gdrive](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link), unzip it, and save it to `data/nuscenes/occ3d`.

(2) Generate the panoptic occupancy ground truth with `gen_instance_info.py`. The panoptic version of Occ3D will be saved to `data/nuscenes/occ3d_panoptic`.

step 4. CKPTS Preparation
(1) Download flashocc-r50-256x704.pth[https://drive.google.com/file/d/1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B/view] to Path_to_FlashOcc/FlashOcc/ckpts/, then run:

```shell
bash tools/dist_test.sh projects/configs/flashocc/flashocc-r50.py  ckpts/flashocc-r50-256x704.pth 4 --eval map
```

step 5. (Optional) Install mmdeploy for tensorrt testing

```shell
conda activate FlashOcc
pip install Cython==0.29.24

### get tensorrt
wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/8.4.0/tars/TensorRT-8.4.0.6.Linux.x86_64-gnu.cuda-11.6.cudnn8.3.tar.gz
export TENSORRT_DIR=Path_to_TensorRT-8.4.0.6

### get onnxruntime
ONNXRUNTIME_VERSION=1.8.1
pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION}
cd Path_to_your_onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
     && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
# export ONNXRUNTIME_DIR=/data01/shuchangyong/pkgs/onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=Path_to_your_onnxruntime/onnxruntime-linux-x64-1.8.1
cd Path_to_FlashOcc/FlashOcc/
git clone git@github.com:drilistbox/mmdeploy.git
cd Path_to_FlashOcc/FlashOcc/mmdeploy
git submodule update --init --recursive
mkdir -p build
cd Path_to_FlashOcc/FlashOcc/mmdeploy/build
cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" ..
make -j 16
cd Path_to_FlashOcc/FlashOcc/mmdeploy
pip install -e .

### build sdk
cd Path_to_pplcv/
git clone https://github.com/openppl-public/ppl.cv.git
cd Path_to_pplcv/ppl.cv
export PPLCV_VERSION=0.7.0
git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION}
./build.sh cuda

#pip install nvidia-tensorrt==8.4.0.6
pip install nvidia-tensorrt==8.4.1.5
pip install tensorrt
#pip install h5py
pip install spconv==2.3.6

export PATH=Path_to_TensorRT-8.4.0.6/bin:$PATH
export LD_LIBRARY_PATH=Path_to_TensorRT-8.4.0.6/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=Path_to_TensorRT-8.4.0.6/lib:$LIBRARY_PATH
```

## The finally overall rangement

1. Tensort

```shell
└── Path_to_TensorRT-8.4.0.6
    └── TensorRT-8.4.0.6
```

2. FlashOcc

```shell
└── Path_to_FlashOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval (existing)
            ├── sweeps  (existing)
            ├── samples (existing)
            ├── gts (new)
            ├── bevdetv2-nuscenes_infos_train.pkl (new)
            └── bevdetv2-nuscenes_infos_val.pkl (new)
    └── doc
        ├── install.md
        └── trt_test.md
    ├── figs
    ├── mmdeploy (new)
    ├── mmdetection3d (new)
    ├── projects
    ├── requirements
    ├── tools
    └── README.md
```

3. ppl.cv

```shell
└── Path_to_pplcv
    └── ppl.cv
```
