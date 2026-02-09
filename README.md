# CleanFlow 🧹

**CleanFlow** is a modular Python library designed to automate the most tedious parts of data cleaning using Software Engineering (SWE) best practices. It transforms messy datasets into analysis-ready dataframes with a single pipeline.

## 🚀 Features
- **Automated Type Inference**: Automatically detects and fixes date strings and currency formats.
- **Smart Imputation**: Uses scikit-learn logic to handle missing values for both numeric and categorical data.
- **Outlier Handling**: Detects and caps extreme values using the Interquartile Range (IQR) method.
- **Quality Reporting**: Generates "Before and After" health reports (completeness, memory usage, duplicates).

## 🛠 Installation
Clone the repo and install locally:
```bash
git clone [https://github.com/lcabrp/cleanflow.git](https://github.com/lcabrp/cleanflow.git)
cd cleanflow
pip install .