from setuptools import find_packages
from setuptools import setup

install_requires = [
    'torch>=1.5.1',
    'gym>=0.17.2',
    'numpy',
    'pillow',
    'optuna',
    'cloudpickle==1.3.0',
    'cycler==0.10.0',
    'future==0.18.2',
    'kiwisolver==1.2.0',
    'matplotlib',
    'pandas',
    'pyglet==1.5.0',
    'pyparsing==2.4.7',
    'python-dateutil==2.8.1',
    'pytz==2020.1',
    'scipy',
    'seaborn',
    'six',
    'tabulate==0.8.7',
]

setup(
    name='pic',
    version='0.0.1',
    description='',
    author='Hiroki Furuta',
    author_email='',
    url='',
    license='MIT License',
    packages=find_packages(),
    install_requires=install_requires,
)
