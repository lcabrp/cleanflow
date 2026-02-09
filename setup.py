from setuptools import setup, find_packages

setup(
    name="cleanflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "numpy",
    ],
    author="Leonardo Abreu",
    description="A modular data cleaning pipeline following SWE best practices.",
)