from setuptools import setup, find_packages

setup(
    name="cleanflow",
    version="0.3.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "parquet": ["pyarrow>=10"],
        "duckdb": ["duckdb>=0.10"],
        "polars": ["polars>=0.20"],
        "dask": ["dask[dataframe]>=2023.1"],
        "all": ["pyarrow>=10", "duckdb>=0.10", "polars>=0.20", "dask[dataframe]>=2023.1"],
        "dev": ["pytest>=7", "ruff>=0.4"],
    },
    entry_points={
        "console_scripts": [
            "cleanflow-optimize=cleanflow.cli:optimize_main",
            "cleanflow-profile=cleanflow.cli:profile_main_entry",
        ],
    },
    author="Leonardo Abreu",
    description="Data cleaning, quality analysis, profiling, and optimization utilities",
)
