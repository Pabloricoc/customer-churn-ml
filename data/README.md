# Data

This directory contains the datasets used in the Customer Churn Prediction project.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

Each row represents a telecommunications customer and contains demographic,
service, contract, billing and churn information.

The dataset contains 7,043 customer observations and 21 variables.

## Source

Dataset: **IBM Telco Customer Churn**

The dataset is available from IBM's public sample repository:

https://github.com/IBM/watsonx-ai-samples/blob/master/cpd4.5/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv

Expected filename:

`WA_FnUseC_TelcoCustomerChurn.csv`

After downloading the file, place it in:

`data/raw/WA_FnUseC_TelcoCustomerChurn.csv`

Raw data files are intentionally excluded from Git version control.

## Directory structure

```text
data/
├── raw/
│   └── WA_FnUseC_TelcoCustomerChurn.csv
└── processed/