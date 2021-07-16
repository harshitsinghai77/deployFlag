"""Module to train the ML model for each repo."""

import pickle
import time

from pandas import CategoricalDtype
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from deployflag.core.constants import (
    FREQ_DETECTED_QUANTILE,
    ISSUE_CATEGORY,
    ML_CLASSIFIER_EVAL_METRIC,
    ML_CLASSIFIER_OBJECTIVE,
    ML_GRID_SEARCH_HYPERPARAMETER,
)
from deployflag.core.utils import create_dummy_cols
from deployflag.logger import LOGGER


class ModelTrainer:
    """
    1. This is the training module of deployflag.
    2. To train ML models for each different repo with all the issues flagged from the begining

    Input -
    1. df - Dataframe containing features Data having following columns -
    repo_id
    issue_id
    timestamp
    frequency_detected
    type
    severity_score
    days_after_first_seen
    days_after_last_seen
    days_after_last_fixed
    freq_detected_quantile
    last_fix_rate
    last_2_fix_rate
    month
    quarter
    fix_rate_binary

    2. repo_id - Repo Id
    3. save_model - Boolean flag to save the trained model pickle file on disk
    4. model_path - Disk Path to store the model pickle file

    Output -
    1. Roc score (roc_score)
    2. Precision score (precision_score)
    3. Training time (total_training_time)
    4. Trained ML Model (xg_clf)

    Usage -
    1. Trained models return the probability of fixing next occurrences of issue
       based on user's past behaviour
    """

    def __init__(self, df, repo_id, save_model=False, model_path=None):
        self.df = df
        self.repo_id = repo_id
        # by default false (not saving the model on disk)
        self.save_model = save_model
        self.model_path = model_path

    def fit(self):
        """Train a classifier using the training data."""
        feature_data = self.df

        y = feature_data[["fix_rate_binary"]]

        # Create dummy cols for object type columns - Training models don't take object columns
        feature_data["quarter"] = feature_data["quarter"].astype(
            CategoricalDtype([1, 2, 3, 4])
        )
        feature_data["month"] = feature_data["month"].astype(
            CategoricalDtype([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        )
        feature_data["freq_detected_quantile"] = feature_data[
            "freq_detected_quantile"
        ].astype(CategoricalDtype(FREQ_DETECTED_QUANTILE))
        feature_data["days_after_last_fixed"] = feature_data[
            "days_after_last_fixed"
        ].astype(CategoricalDtype(["Never", "Short", "Long"]))
        feature_data["type"] = feature_data["type"].astype(
            CategoricalDtype(ISSUE_CATEGORY)
        )

        feature_data = create_dummy_cols(
            feature_data,
            [
                "issue_id",
                "quarter",
                "month",
                "freq_detected_quantile",
                "days_after_last_fixed",
                "type",
            ],
        )

        # Drop redudant variables from training dataset
        X = feature_data.drop(
            ["repo_id", "fix_rate_binary"],
            axis=1,
        )

        # starting time for model training
        training_time_start = time.time()

        # Perform Gridsearch to tune the hyperparameters
        # Extend methods to tune on larger sets in the server
        estimator = XGBClassifier(
            objective=ML_CLASSIFIER_OBJECTIVE,
            eval_metric=ML_CLASSIFIER_EVAL_METRIC,
            use_label_encoder=False,
            seed=42,
        )

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=ML_GRID_SEARCH_HYPERPARAMETER,
            scoring="roc_auc",
            n_jobs=1,
            cv=2,
            verbose=False,
        )

        grid_search.fit(X, y)

        # Select the best estimator
        xg_clf = grid_search.best_estimator_
        xg_clf.fit(X, y)

        # total time for model training
        total_training_time = time.time() - training_time_start

        y_pred = xg_clf.predict(X)

        # Calculate ROC score and precision
        output_roc_score = roc_auc_score(y, y_pred)
        output_precision_score = precision_score(
            y, y_pred, average="binary", zero_division=1
        )

        xg_clf.feature_names = list(X.columns)

        if self.save_model and self.model_path:
            # Save the model on disk (by default False)
            model_name = f"model_{str(self.repo_id)}.pkl"
            filename = f"{self.model_path}/{model_name}"
            pickle.dump(xg_clf, open(filename, "wb"))

        return output_roc_score, output_precision_score, total_training_time, xg_clf
