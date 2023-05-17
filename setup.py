from setuptools import setup, find_packages

setup(
    name="torch_activation",
    version="0.1",
    author="Alan Huynh",
    author_email="hdmquan@gmail.com",
    description="A collection of new activation functions for PyTorch",
    url="https://github.com/alan191006/C-FCN-PyTorch-implementation",
    packages=find_packages(),
    install_requires=[
        "torch",    
    ],
)