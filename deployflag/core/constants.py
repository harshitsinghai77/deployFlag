DEPLOY_FLAG_VERSION = "0.1.0"

# Change the ML_MODEL_VERSION when changing other configuration to sync it with the database.
ML_MODEL_VERSION = "v1.0"
ML_CLASSIFER_NAME = "xgboost.XGBClassifier"
ML_CLASSIFIER_VERSION = "1.3.3"

# Change the ML_GRID_SEARCH_PARAMETER_VERSION when changing other configuration to sync it with the database.
# Forgetting to change the VERSION while making other changes will result in ambiguity in metadata db
# and lead to corrupted metadata sync.
ML_GRID_SEARCH_PARAMETER_VERSION = "v1.0"
ML_CLASSIFIER_OBJECTIVE = "binary:logistic"
ML_CLASSIFIER_EVAL_METRIC = "logloss"
ML_GRID_SEARCH_HYPERPARAMETER = {
    "max_depth": [10],
    "n_estimators": [600],
    "learning_rate": [0.01],
    "gamma": [1],
    "colsample_bytree": [0.5],
}

# This can be change iresspective of both MODEL_VERSION and GRID_SEARCH_PARAMETER_VERSION
# i.e changes to this doesn't need chaninging the above versions.

# Note: Sum of w1, w2, w3, w4 would be equal to 1.
W1_DAILY_WEIGHTED_FIX_RATE = 0.3  # Weighted Fix Rate (Daily)
W2_HISTORICAL_WEIGHTED_FIX_RATE = 0.3  # Weighted Historical Fix Rate
W3_FIX_PROBABILITY = 0.2  # Probability of Fixing the Issue at particular point of time
W4_SEVERITY = 0.2  # Severity

ISSUE_CATEGORY = [
    "antipattern",
    "bug-risk",
    "performance",
    "security",
    "style",
    "doc",
    "typecheck",
    "coverage",
]

FREQ_DETECTED_QUANTILE_LOW = "low"
FREQ_DETECTED_QUANTILE_MEDIUM_LOW = "medium_low"
FREQ_DETECTED_QUANTILE_MEDIUM_HIGH = "medium_high"
FREQ_DETECTED_QUANTILE_HIGH = "high"

FREQ_DETECTED_QUANTILE = [
    FREQ_DETECTED_QUANTILE_LOW,
    FREQ_DETECTED_QUANTILE_MEDIUM_LOW,
    FREQ_DETECTED_QUANTILE_MEDIUM_HIGH,
    FREQ_DETECTED_QUANTILE_HIGH,
]
