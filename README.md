# Customer Churn Prediction

End-to-end machine learning project for predicting customer churn in a telecommunications company.

The project follows a structured data science workflow, from data understanding and exploratory analysis to reproducible preprocessing, model validation, model comparison and final evaluation.

## Business Problem

Customer churn represents customers who stop using a company's services.

The objective of this project is to estimate which customers are at higher risk of churn so that a company could potentially prioritize retention actions.

This is formulated as a binary classification problem:

- `Churn = Yes`: the customer left the company.
- `Churn = No`: the customer remained with the company.

The project focuses not only on predictive performance, but also on reproducibility, data leakage prevention, appropriate evaluation metrics and model interpretability.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains:

- **7,043 customer observations**
- **21 original variables**
- customer demographic information;
- subscribed services;
- contract information;
- billing and payment information;
- customer tenure;
- churn status.

The target distribution is moderately imbalanced:

- approximately **73.5% non-churn customers**
- approximately **26.5% churn customers**

The raw dataset is intentionally excluded from Git version control.

See [`data/README.md`](data/README.md) for dataset documentation and acquisition instructions.

## Repository Structure

```text
customer-churn-ml/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   └── 03_preprocessing_baseline.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Machine Learning Workflow

### 1. Data Understanding

The initial dataset assessment focused on schema, variable roles and data quality.

Key findings:

- each row represents a unique customer;
- `customerID` is a unique identifier and is excluded from predictive modeling;
- `TotalCharges` is stored as an object although it represents a numerical quantity;
- blank values were identified in `TotalCharges`;
- these problematic records correspond to customers with `tenure = 0`;
- `SeniorCitizen` is numerically encoded but semantically represents a binary categorical feature;
- no completely duplicated customer records were identified.

Notebook:

[`01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb)

### 2. Exploratory Data Analysis

Exploratory analysis was used to identify customer characteristics associated with churn and generate hypotheses for modeling.

Main patterns observed:

- month-to-month customers show substantially higher churn rates than customers with longer contracts;
- churn is more frequent among customers with shorter tenure;
- churned customers tend to have higher monthly charges, although the distributions overlap;
- internet service type is associated with different churn rates;
- electronic check users show relatively high churn;
- customers without online security or technical support show higher churn rates;
- lower `TotalCharges` among churned customers appears strongly related to their shorter tenure;
- tenure and contract type together identify particularly high-risk customer segments.

These findings describe associations in the dataset and should not be interpreted as causal relationships.

Notebook:

[`02_eda.ipynb`](notebooks/02_eda.ipynb)

### 3. Preprocessing

A reproducible preprocessing workflow was implemented using scikit-learn pipelines.

Numerical features:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Numerical preprocessing:

- median imputation;
- standardization using `StandardScaler`.

Categorical preprocessing:

- most-frequent-value imputation;
- one-hot encoding with support for previously unseen categories.

Both preprocessing branches are combined using a `ColumnTransformer`.

The preprocessing pipeline is fitted only using training data in order to reduce the risk of data leakage.

### 4. Train / Test Strategy

The dataset is divided using a stratified 80/20 train/test split.

```text
Training set: 5,634 customers
Test set:     1,409 customers
```

Stratification preserves the original churn distribution.

The test set is treated as held-out data and is intended for final model evaluation rather than repeated model selection.

### 5. Baseline Models

Two initial baselines have been established.

#### DummyClassifier

A majority-class classifier establishes the minimum reference performance.

Although it achieves relatively high accuracy because of the target imbalance, it fails to identify churned customers.

This demonstrates why accuracy alone is insufficient for this problem.

#### Logistic Regression

Logistic Regression is used as the first real machine learning baseline.

It is combined with the complete preprocessing workflow inside a single scikit-learn `Pipeline`.

The model provides meaningful predictive performance beyond the dummy baseline and serves as an interpretable reference for subsequent models.

Notebook:

[`03_preprocessing_baseline.ipynb`](notebooks/03_preprocessing_baseline.ipynb)

## Evaluation Strategy

Because churn is moderately imbalanced, model evaluation does not rely exclusively on accuracy.

The main metrics considered are:

- **ROC-AUC** — ranking ability across classification thresholds;
- **Recall** — proportion of actual churners detected;
- **Precision** — proportion of predicted churners that actually churn;
- **F1-score** — balance between precision and recall;
- **Accuracy** — reported as a secondary metric.

Model development uses **stratified 5-fold cross-validation** on the training set.

This allows models to be compared across multiple data partitions while keeping the final test set isolated.

## Current Project Status

Completed:

- data acquisition and documentation;
- data quality assessment;
- exploratory data analysis;
- feature and target definition;
- stratified train/test split;
- reproducible preprocessing pipeline;
- dummy classification baseline;
- Logistic Regression baseline;
- stratified cross-validation;
- evaluation strategy definition.

In progress / planned:

- Decision Tree baseline;
- Random Forest;
- Gradient Boosting;
- systematic model comparison;
- hyperparameter tuning;
- classification threshold analysis;
- final test-set evaluation;
- model interpretation;
- error analysis;
- refactoring reusable logic into `src/`;
- final portfolio documentation.

## Reproducibility

Create and activate a Python virtual environment, then install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset following the instructions in:

```text
data/README.md
```

Place the raw CSV at:

```text
data/raw/WA_FnUseC_TelcoCustomerChurn.csv
```

The notebooks should then be executed in numerical order:

```text
01_data_understanding.ipynb
02_eda.ipynb
03_preprocessing_baseline.ipynb
```

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Jupyter
- Git
- GitHub

## Project Status

🚧 **Active development**

The next stage of the project focuses on comparing linear, tree-based and ensemble classification models using the same cross-validation framework.S