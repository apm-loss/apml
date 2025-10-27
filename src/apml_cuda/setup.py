from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='apml_sparse',
    ext_modules=[
        CUDAExtension(
            name='apml_sparse',
            sources=['apml_sparse.cpp', 'apml_sparse_kernel.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': [
                    '-g', '-G', '--extended-lambda', '--expt-relaxed-constexpr', '-std=c++17'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)