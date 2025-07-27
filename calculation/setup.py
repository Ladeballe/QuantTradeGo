from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        name='formula_cfuncs',
        sources=['formula_cfuncs.pyx'],
        extra_compile_args=['-O3'],  # Enable optimization flags
    )
]

setup(ext_modules=cythonize(extensions))
