from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='minimal_attn',
    ext_modules=[
        CUDAExtension(
            name='minimal_attn',
            sources=[
                'main.cpp',
                'flash.cu',
                'backward.cu',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)  # Force Ninja
    },

    install_requires=[
        'torch>=2.0.0',
        'ninja',
    ],
    setup_requires=[
        'torch>=2.0.0',
    ],
)
