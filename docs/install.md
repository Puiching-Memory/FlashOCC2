# 安装环境

```bash
* using python=3.13.3 ubuntu2404 cuda12.8.1 docker H800

# pull ubuntu from docker hub (optional)
docker pull ubuntu:24.04

# install python from source (optional)
# doc: https://blog.frognew.com/2024/12/build-python3.13-from-source.html
apt install gcc-13 g++-13 make pkg-config libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev libncursesw5-dev xz-utils liblzma-dev uuid-dev libffi-dev libgdbm-dev tk-dev
cd ./Python-3.13.3/

./configure --enable-optimizations --with-lto --enable-shared
make -j "$(nproc)"
make altinstall

sed -i '$a\export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' ~/.bashrc
source ~/.bashrc

ln -s /usr/local/bin/pip3.13 /usr/local/bin/pip3
ln -s /usr/local/bin/pip3.13 /usr/local/bin/pip
ln -s /usr/local/bin/python3.13 /usr/local/bin/python3
ln -s /usr/local/bin/python3.13 /usr/local/bin/python

# install cuda 12.8 (necessary)
apt install wget gcc-13 g++-13 libxml2
ln -s /usr/bin/gcc-13 /usr/bin/gcc
ln -s /usr/bin/g++-13 /usr/bin/g++
ln -s /usr/bin/g++-13 /usr/bin/c++

wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sh cuda_12.8.1_570.124.06_linux.run

sed -i '$a\export PATH=/usr/local/cuda-12.8/bin:$PATH' ~/.bashrc
sed -i '$a\export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' ~/.bashrc
source ~/.bashrc

# enable parallel compilation (optional)
apt install ninja-build

# enable computational parallel optimization (OpenMP) (optional)
apt install libomp-dev

# fix OpenCV missing lib (necessary)
apt install libgl1-mesa-glx libglu1-mesa mesa-utils
apt install libglx-mesa0 libglu1-mesa mesa-utils # ubuntu2404

# enable unzip *.zip file (optional)
apt install unzip

# enable git tracking (optional)
apt install git

# install python lib (necessary)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install ./3rd_party/nuscenes-devkit/ -v
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.7.0cu128 -v
pip install -r requirements.txt -v
```

### 验证安装:

```
python hello_flashocc2.py
```

# 准备数据

### OpenOcc

我们建议使用单独的python环境安装openxlab，然后在下载完成后删除该环境

https://github.com/OpenDriveLab/OccNet?tab=readme-ov-file#openocc-dataset

你可以选择从opendatalab或googleDrive下载数据集,但是需要注意的是,opendatalab上缺少了nuscenes_infos_test_occ.pkl文件
https://opendatalab.com/OpenDriveLab/CVPR24-Occ-Flow-Challenge
https://drive.google.com/drive/folders/1lpqjXZRKEvNHFhsxTf0MOE13AZ3q4bTq

```bash
openxlab dataset get --dataset-repo OpenDriveLab/CVPR24-Occ-Flow-Challenge --target-path ./dataset/

mv dataset/OpenDriveLab___CVPR24-Occ-Flow-Challenge/*.zip dataset
rm -rf dataset/OpenDriveLab___CVPR24-Occ-Flow-Challenge/
unzip dataset/infos.zip
unzip dataset/openocc_v2.1.zip
mv nuscenes_infos_train_occ.pkl dataset/nuscenes/
mv nuscenes_infos_val_occ.pkl dataset/nuscenes/
mv openocc_v2 dataset/nuscenes/
rm dataset/infos.zip
rm dataset/openocc_v2.1.zip
```

最终，文件结构如下:
```
nuscenes
├── maps
├── nuscenes_infos_train_occ.pkl
├── nuscenes_infos_val_occ.pkl
├── nuscenes_infos_test_occ.pkl
├── openocc_v2
├── samples
├── v1.0-test
└── v1.0-trainval
```
