# setup.py for the SOLVE library
from setuptools import setup, find_packages

setup(
    name='solve_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'pandas',
        'joblib'
    ],
    author='Saikat',
    author_email='cssd2399@iacs.res.in',
    description='A package for extracting subsequences from any sequences.',
    url='https://github.com/saikat-ai/Enzyme_prediction/solve_package',
    license='MIT'
)
