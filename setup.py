from setuptools import setup


setup(
    cffi_modules=["src/pyllsm2/_build.py:ffibuilder"],
)

