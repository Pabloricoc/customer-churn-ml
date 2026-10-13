"""Project-wide configuration constants."""

DATA_FILENAME = "WA_FnUseC_TelcoCustomerChurn.csv"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42

FINAL_THRESHOLD = 0.35

FINAL_MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 1,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
}