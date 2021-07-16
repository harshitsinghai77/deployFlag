"""Module to calculate the final metric for each issue."""

import pandas as pd

from deployflag.core.utils import check_columns, movecol, weighted_average
from deployflag.logger import LOGGER


class MetricCalculator:
    """
    This is output point of deployflag which would return score for each repo and issue.
    Loads required datasets and training model to calculate scores basis weights

    This class uses the preprocessed data (given by the preprocessing module) containing the fix rate of an issue
    and some created fixtures which are required for metric calculation.

    Data Required for initializing the class:
    1. `preprocessed_data`: Data of an issue having details:
    repo_id
    issue_id
    timestamp
    frequency_detected
    type
    severity_score
    days_after_first_seens
    days_after_last_seen
    days_after_last_fixed
    freq_detected_quantile
    last_fix_rate
    last_2_fix_rate
    month
    quarter
    fix_rate_binary

    2. `fr_data`:  Data with created features. This has the following column:
    repo_id
    issue_id
    timestamp
    analysis_type
    frequency_detected
    fix_rate

    3. `model_store_path`: path to the pickle file of the trained model for this repo
    4. `repo`: repo id
    5-8: Weights for each different parameter viz. Historical Weighted FR,
    Daily Weighted FR, Fix Probablity and Internal Severity
    """

    def __init__(
        self,
        fr_data,
        preprocessed_data,
        repo,
        xg_model,
        model_store_path=None,
        w1=0.3,
        w2=0.3,
        w3=0.2,
        w4=0.2,
    ):
        self.fr_data = fr_data
        self.preprocessed_data = preprocessed_data
        self.repo = repo
        self.xg_model = xg_model
        self.model_store_path = model_store_path
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def transform(self):
        """Calculate the final metric of an issue for each repo."""
        fr_data, processed_data, model = (
            self.fr_data,
            self.preprocessed_data,
            self.xg_model,
        )

        # If model not extracted properly return 0 probability
        if model:
            # Extract features which were used to train model
            features_required = model.feature_names

            # Check if issue exists in features or not, if not then return fix probablity = 0
            if "issue_id__" + str(fr_data["issue_id"][0]) in features_required:

                test_data = processed_data

                # Check if the required features are present in the data or not
                # Extracting the unique features out of the model which were used during training
                unique_features = []
                for feature in features_required:
                    if "__" in feature:
                        unique_features.append(feature.split("__")[0])
                    else:
                        unique_features.append(feature)

                # Taking the unique out of the unique features
                unique_features = list(set(unique_features))

                if not check_columns(test_data, unique_features):
                    col_names = []
                    for col in unique_features:
                        if col not in list(test_data.columns):
                            col_names.append(col)

                    LOGGER.error(
                        "Columns missing from test data %s", col_names)

                    raise TypeError(
                        "Columns missing from test data ", col_names)

                # Slicing the object columns only to create dummy columns
                # Create dummy cols for object type columns - Training models don't take object columns
                cols_to_create = features_required[6:]

                # Initializing dummy cols with zero
                for col in cols_to_create:
                    test_data = test_data.assign(**{col: 0})

                # Filling the 1s in the object columns where issue has the value
                # this is required because test data has been created from processed data on the fly within this module
                # and test data would require to contain all the dummy columns which were there in training data
                # That too only for final row - because probablity of fix rate
                # would be calculated for only most recent data point
                for col in [
                    "issue_id",
                    "month",
                    "quarter",
                    "freq_detected_quantile",
                    "days_after_last_fixed",
                    "type",
                ]:
                    val = processed_data[col][len(test_data) - 1]
                    col_name = col + "__" + str(val)
                    test_data.loc[len(test_data) - 1, col_name] = 1

                # Drop columns which aren't required and were not there while training the model
                # columns in which values are constant or constantly changing (id cols, timestamps)
                test_data = test_data.drop(
                    [
                        "repo_id",
                        "fix_rate_binary",
                        "timestamp",
                        "issue_id",
                        "month",
                        "quarter",
                        "freq_detected_quantile",
                        "days_after_last_fixed",
                        "type",
                    ],
                    axis=1,
                )

                # Re-ordering the columns - model takes ordered features -
                # This happened because we initialised new dummy cols
                test_data = movecol(
                    test_data,
                    cols_to_move=["severity_score"],
                    ref_col="last_2_fix_rate",
                )

                # Use of tail for recent input data
                # Calculate probability of fixing the most recently detected occurrences of issue
                fix_probab = model.predict_proba(test_data.tail(1))[:, 1]
                fix_probab = fix_probab[0]

            else:
                LOGGER.info(
                    "Model hasn't trained for this issue %s", fr_data["issue_id"][0]
                )
                LOGGER.info("Setting fix_probab to 0")
                fix_probab = 0

        else:
            fix_probab = 0

        fr_data["timestamp"] = pd.to_datetime(
            fr_data["timestamp"], format="%Y-%m-%dT%H:%M:%S.%fZ"
        )
        fr_data = fr_data.sort_values(by=["timestamp"])

        processed_data["timestamp"] = pd.to_datetime(
            processed_data["timestamp"], format="%Y-%m-%dT%H:%M:%S.%fZ"
        )
        processed_data = processed_data.sort_values(by=["timestamp"])

        dict_to_return = {}

        # Calculate daily weighted average for each issue by passing the filtered series
        daily_weighted_fix_rate = weighted_average(
            list(fr_data[fr_data.analysis_type == "daily"]["fix_rate"])
        )

        # Calculate historical weighted average for each issue by passing the filtered series
        historical_weighted_fix_rate = weighted_average(
            list(fr_data[fr_data.analysis_type == "historical"]["fix_rate"])
        )

        # Record severity of the issue with test_data's most recent data
        severity = processed_data["severity_score"][len(processed_data) - 1]

        # Scaling the severity between 0-1 because all other parameters are in range 0-1
        severity = severity / 100

        # Calculate final metric by multiplying weights and values
        final_score = (
            (self.w1 * daily_weighted_fix_rate)
            + (self.w2 * historical_weighted_fix_rate)
            + (self.w3 * fix_probab)
            + (self.w4 * severity)
        )

        # Check if score is greater than 1
        final_score = round(final_score * 100,
                            0) if (final_score * 100) <= 100 else 100

        dict_to_return["issue_shortcode"] = fr_data["issue_id"][0]
        dict_to_return["issue_weight"] = int(final_score)

        return dict_to_return
