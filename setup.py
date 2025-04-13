from setuptools import find_packages, setup
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

def make_cuda_ext(
    name, module, sources, sources_cuda=[], extra_args=[], extra_include_path=[]
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        print(f"Compiling {name} with CUDA")
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda
    else:
        print(f"Compiling {name} without CUDA")
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="flashocc2",
        ext_modules=[
            make_cuda_ext(
                name="bev_pool_ext",
                module="flashocc2.ops.bev_pool",
                sources=[
                    "src/bev_pooling.cpp",
                    "src/bev_sum_pool.cpp",
                    "src/bev_sum_pool_cuda.cu",
                    "src/bev_max_pool.cpp",
                    "src/bev_max_pool_cuda.cu",
                ],
                extra_include_path=[
                    "src",
                ],
            ),
            make_cuda_ext(
                name="bev_pool_v2_ext",
                module="flashocc2.ops.bev_pool_v2",
                sources=["src/bev_pool.cpp", "src/bev_pool_cuda.cu"],
                extra_include_path=[
                    "src",
                ],
            ),
            make_cuda_ext(
                name="nearest_assign_ext",
                module="flashocc2.ops.nearest_assign",
                sources=["src/nearest_assign.cpp", "src/nearest_assign_cuda.cu"],
                extra_include_path=[
                    "src",
                ],
            ),
            make_cuda_ext(
                name="occ_pool_ext",
                module="flashocc2.ops.occ_pool",
                sources=["src/occ_pool.cpp", "src/occ_pool_cuda.cu"],
                extra_include_path=[
                    "src",
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )
