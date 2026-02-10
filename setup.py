from setuptools import setup, find_packages

setup(
    name="cleanflow",
    version="0.2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
    ],
    author="Leonardo Abreu",
    description="Modular Python library for automated data cleaning",
)