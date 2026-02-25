from setuptools import find_packages, setup

import os
import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)



if __name__ == '__main__':
    setup(
        ext_modules=[
            CUDAExtension(
                name="bev_pool_v2_ext",
                sources=[
                    "src/C/bev_pool.cpp",
                    "src/C/bev_pool_cuda.cu"
                ],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        )