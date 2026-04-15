# online-shoppers-tools

![Tests](https://github.com/UBC-DSCI-310-2025W2/online-shoppers-tools/actions/workflows/tests.yml/badge.svg)

A lightweight Python package for data preprocessing, exploratory analysis, and model evaluation for online shopping behavior datasets(Sakar, C & Kastro Y., 2018), or other dataset with similar format.

## Overview

`online-shoppers-tools` is a reusable Python package designed to support common tasks in data science workflows, particularly for classification problems of the online shopping datasets.

The package provides functions for:

- cleaning and transforming data
- converting boolean-like variables
- analyzing class balance
- generating feature importance plots
- evaluating classification models with performance metrics

These utilities were originally developed as part of an analysis pipeline and have been modularized into a standalone package to promote reuse and reproducibility.

## Repo structure

```
online-shoppers-tools/
├── src/
│   └── online_shoppers_tools/
│       ├── __init__.py
│       ├── calculate_class_balance.py
│       ├── convert_boolean_values.py
│       ├── create_feat_importance_plot.py
│       └── evaluate_model.py
├── tests/
├── pyproject.toml
├── conda-lock.yml
├── environment.yml
├── README.md
```

## Dependencies

The package replies on several core libraries defined in pyproject.toml

* pandas
* numpy
* scikit-learn
* matplotlib

## Position in the Ecosystem

This package sits within the Python data science ecosystem alongside tools such as:

- pandas — for data manipulation
- scikit-learn — for machine learning models and metrics
- matplotlib — for visualization

Unlike these general-purpose libraries, online-shoppers-tools provides task-specific helper functions tailored for:

- classification workflows for the online_shoppers dataset
- quick evaluation pipelines
- reusable analysis steps

It acts as a glue layer between these libraries, simplifying repetitive tasks.

## Installation

This package can be used in two ways: with **Conda** via `environment.yml`, or with **Hatch**.

The environment.yml file creates a Conda environment with the required dependencies. Hatch, on the other hand, manages package-specific workflows such as testing and building. The two tools are complementary: environment.yml helps set up the environment, while Hatch helps manage the package.

### Option 1: Using `environment.yml` (Conda environment)

If you would like to create a reproducible Conda environment first, run:

```bash
conda env create -f environment.yml
conda activate shopperspkg
#Then install package locally using
pip install -e .
```

### Option 2  Using Hatch

After cloning this repo, run

```bash
hatch run test:run
```

to execute the test, or use

```bash
hatch build
```

to build the package

## Usage

```python
from online_shoppers_tools import (
    calculate_class_balance,
    convert_boolean_columns,
    create_feat_importance_plot,
    evaluate_model,
)
# Convert boolean columns to integers
df = convert_boolean_columns(df, ["Weekend", "SpecialDay"])

# Examine class balance
balance = calculate_class_balance(df, target_col="Revenue")
print(balance)

# Train a simple model
X = df.drop(columns="Revenue")
y = df["Revenue"]
model = RandomForestClassifier(random_state=123)
model.fit(X, y)

# Evaluate the model
results = evaluate_model(model, X, y)
print(results)

# Plot feature importance
create_feat_importance_plot(model, X, max_display=2)
```

## References

Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q.