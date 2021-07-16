"""Module for preprocessing the raw data."""

import numpy as np
import pandas as pd

from deployflag.core.exceptions import NoRowsCreatedWhilePreprocessing
from deployflag.core.utils import calculate_quantiles, check_columns, if_all_zero, is_same


def create_features(final, issue_type, severity, last_fixed_duration):
    """
    Convert fix rate data, issues type, severity to create 8 derived features and 1 output label.
    Which would be required for modelling as well as metric calculation.

    Input Data-
    Fix rate data, issues type, severity, Duration to be set for categorising days after last fixed

    Output Data -
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

    Usage -
    Input to Modelling and Metric Calculation modules
    """
    # Checking the required columns for each data
    final_cols = ["repo_id", "issue_id",
                  "timestamp", "analysis_type", "fix_rate"]
    if not check_columns(final, final_cols):
        raise TypeError(
            f"Some columns are missing. Expected columns: {final_cols}")

    # Convert string datetime to python readable datetime format
    final.loc[:, "timestamp"] = pd.to_datetime(
        final["timestamp"], format="%Y-%m-%dT%H:%M:%S.%fZ"
    )

    # Sort the values by datetime to traverse data chronologically
    final = final.sort_values(by=["timestamp"])

    # Assign Values
    final.loc[:, "type"] = issue_type
    final.loc[:, "severity_score"] = severity

    # Initialize new features with NA values
    final = final.assign(
        **{
            "days_after_first_seen": np.nan,
            "days_after_last_seen": np.nan,
            "days_after_last_fixed": np.nan,
            "freq_detected_quantile": np.nan,
            "last_fix_rate": np.nan,
            "last_2_fix_rate": np.nan,
        }
    )

    final.loc[:, "month"] = final["timestamp"].dt.month
    final.loc[:, "quarter"] = final["timestamp"].dt.quarter

    # Capture first_date to calculate - days after first seen
    first_date = final.iloc[0]["timestamp"]

    # Initialize last_fix_date so as to calculate days
    last_fix_date = ""
    COLUMNS = [
        "days_after_last_seen",
        "last_fix_rate",
        "last_2_fix_rate",
    ]
    final_indexes = list(final.index)

    def create_features_days_after_last_fixed(row):
        """Return days_after_last_fixed categorical value."""
        fix_rate, timestamp = row[0], row[1]
        nonlocal last_fix_date
        if last_fix_date == "":
            local_days_after_last_fixed = "Never"
        else:
            days = (timestamp - last_fix_date).days
            # If difference of days after last fixed is greater than last fixed duration then
            # IT's a long time since user has fixed the issue
            # Else It's just short duration
            if days >= last_fixed_duration:
                local_days_after_last_fixed = "Long"
            else:
                local_days_after_last_fixed = "Short"

        if fix_rate > 0:
            last_fix_date = timestamp

        return local_days_after_last_fixed

    final["days_after_first_seen"] = final["timestamp"].apply(
        lambda x: (x - first_date).days
    )
    final["freq_detected_quantile"] = final["frequency_detected"].apply(
        lambda x: calculate_quantiles(final.frequency_detected, x)
    )

    if if_all_zero(final["fix_rate"]):
        # If fix rate is zero for each appearance then user never fixed the issue
        final["days_after_last_fixed"] = "Never"
    else:
        final["days_after_last_fixed"] = final[["fix_rate", "timestamp"]].apply(
            create_features_days_after_last_fixed, axis=1
        )

    # iterate over each index
    for row in final.itertuples():
        index = row.Index
        # Modify values in main dataframe in feature column at the same index

        if index == final.index[0]:
            # For first appearance of issue - last seen difference would be zero
            days_after_last_seen = 0
            last_date = row.timestamp
        else:
            # Calculates difference of days between last seen and current seen
            days_after_last_seen = (row.timestamp - last_date).days

        # Update last_date at each instance
        last_date = row.timestamp

        # Capture the lag of fix rate - so fix rate at t-1 and t-2 instances
        # This would capture the time dependence user behaviour - basically trend of fix rate
        # Assumption - If user has fixed issues in previous instances - it may affect the next fix rate
        if index == final.index[0]:
            last_fix_rate = 0
            last_2_fix_rate = 0
        else:
            last_fix_rate = final.fix_rate[final.index[final_indexes.index(
                index) - 1]]
            if index == final.index[1]:
                last_2_fix_rate = 0
            else:
                last_2_fix_rate = final.fix_rate[
                    final.index[final_indexes.index(index) - 2]
                ]

        final.loc[index, COLUMNS] = [
            days_after_last_seen,
            last_fix_rate,
            last_2_fix_rate,
        ]

    # Create dependent variable for training the model while calculating metric
    # fix_rate_binary would be the probability output
    final["fix_rate_binary"] = final["fix_rate"].apply(
        lambda x: 1 if x > 0 else 0)

    final = final.drop(
        [
            "fix_rate",
            "analysis_type",
        ],
        axis=1,
    )
    return final


class FixRateCalculator:
    """
    1. This is the input point of deployflag data preprocessing.
    2. As mentioned in the spec doc, to decide the relevancy - Issues resolved would define the user's past behaviour
    towards any particular
    3. This module calculates all the instances of issues detected for a repo as well as tracks the issues which got
    resolved with a mathematical logic described in spec doc

    Input Data -
    1. `daily_issue_df` - Analysis Runs data having following columns -
    repository_id | run_id | branch_name | base_oid - Hash Id (String) | commit_oid | check_id
    created_at | shortcode | frequency_detected | frequency_resolved

    2. `historical_issue_df` - Historical Runs data having following columns
    run_id - Numeric
    base_oid - Hash Id (String)
    timestamp - Datetime - Standard Format
    shortcode - String
    frequency - Int

    3. `repo` - Repo Id
    4. `issue_type` - Anti-pattern, security etc.
    5. `severity` - Severity score of Issue
    6. `last_fixed_duration` - Duration which categories days after last fixed in Short, Long Period

    Output Data-
    1. Fix Rate Data - for each issue at each base_oid for Analysis Runs and Historical Runs having following columns
    repo_id
    issue_id
    timestamp
    analysis_type
    frequency_detected
    fix_rate

    2. Features Data having following columns -
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

    Usage -
    1. This output data would become input to Training module to train the models on scheduled basis
    2. Also, for calculating metric scores after each trigger - like after each analysis run etc.
    """

    def __init__(
        self,
        daily_issue_df,
        historical_issue_df,
        repo_id,
        issue_type,
        severity,
        last_fixed_duration=20,
    ):
        self.daily_issue_df = daily_issue_df
        self.historical_issue_df = historical_issue_df
        self.repo_id = repo_id
        self.issue_type = issue_type
        self.severity = severity
        self.last_fixed_duration = last_fixed_duration
        self.rows_list = []

    def perform_historical_analysis(self, df):
        """Calculate fix rate for each historical decline in frequency of issue."""
        shortcode = df["shortcode"][0]

        if is_same(df["frequency"]):
            # Assign fix rate = 0 if all values of frequencies are same - Concluding user hasn't fixed any issue
            self.rows_list.append(
                {
                    "repo_id": self.repo_id,
                    "issue_id": shortcode,
                    "timestamp": df["timestamp"][len(df) - 1],
                    "analysis_type": "historical",
                    "frequency_detected": df["frequency"][0],
                    "fix_rate": 0,
                }
            )
        else:
            for row in df.iloc[:-1, :].itertuples():
                index = row.Index
                # Run loop till the last index of df starting from 0 to length-2 to preserve count+1
                if row.frequency > df["frequency"][index + 1]:
                    # If the frequency declines meaning user has
                    # fixed issues from master branch - Record and calculate Fix Rate
                    detected = row.frequency  # How many detected
                    resolved = (
                        row.frequency - df["frequency"][index + 1]
                    )  # And How many of frequency detected user resolved
                    fixRate_i = round(resolved / detected, 4)
                    self.rows_list.append(
                        {
                            "repo_id": self.repo_id,
                            "issue_id": shortcode,
                            "timestamp": df["timestamp"][index + 1],
                            "analysis_type": "historical",
                            "frequency_detected": detected,
                            "fix_rate": fixRate_i,
                        }
                    )

    def perform_daily_analysis(self, df, issue_id):
        """Calculate fix rate for each issue and for each unique base_oid in branches other than master."""
        # Capture the initial frequency detected
        detected = df["frequency_detected"][0]
        timestamp = df["created_at"][len(df) - 1]
        resolved = 0
        if is_same(df["frequency_detected"]):
            # Assign fix rate = 0 if all values of frequencies are same - Concluding user hasn't fixed any issue
            self.rows_list.append(
                {
                    "repo_id": self.repo_id,
                    "issue_id": issue_id,
                    "timestamp": timestamp,
                    "analysis_type": "daily",
                    "frequency_detected": detected,
                    "fix_rate": 0,
                }
            )
        else:
            for row in df.iloc[:-1, :].itertuples():
                # Run loop till the last index of temp df starting from 0 to length-2 to preserve count+1
                index = row.Index
                # if count != (len(df) - 1):
                # If frequencies are same then no change in new detected
                if row.frequency_detected == df["frequency_detected"][index + 1]:
                    detected = detected + 0
                # If frequency of issue decreased then user resolved some issues
                # increase resolved count by delta
                if row.frequency_detected > df["frequency_detected"][index + 1]:
                    resolved = resolved + (
                        row.frequency_detected -
                        df["frequency_detected"][index + 1]
                    )
                # If frequency of issues increases then new instances of issues got introduced
                # increase detected count by delta
                if row.frequency_detected < df["frequency_detected"][index + 1]:
                    detected = detected + (
                        df["frequency_detected"][index + 1] -
                        row.frequency_detected
                    )
            # Append the list by fix rate by calculating resolved/detected
            self.rows_list.append(
                {
                    "repo_id": self.repo_id,
                    "issue_id": issue_id,
                    "timestamp": timestamp,
                    "analysis_type": "daily",
                    "frequency_detected": detected,
                    "fix_rate": round(resolved / detected, 4),
                }
            )

    def process_historical_data(self):
        """Process dataframe for historical_analysis."""
        historical_issue = self.historical_issue_df
        # Check required columns in historical_issue_df
        hist_cols = [
            "run_id",
            "base_oid",
            "timestamp",
            "shortcode",
            "frequency",
        ]
        if not check_columns(historical_issue, hist_cols):
            raise TypeError(f"Columns are missing. Expected: {hist_cols}")

        # Convert the datetime into python readable datetime format
        historical_issue.loc[:, "timestamp"] = pd.to_datetime(
            historical_issue["timestamp"], format="%Y-%m-%dT%H:%M:%S.%f"
        )

        # Sort the data basis time values so the traversal happens chronologically
        historical_issue = historical_issue.sort_values(by=["timestamp"])
        historical_issue.reset_index(drop=True, inplace=True)

        self.perform_historical_analysis(historical_issue)

    def process_daily_data(self):
        """Process dataframe for daily_analysis."""
        daily_issue = self.daily_issue_df
        daily_cols = [
            "repository_id",
            "run_id",
            "branch_name",
            "base_oid",
            "commit_oid",
            "check_id",
            "created_at",
            "shortcode",
            "frequency_detected",
            "frequency_resolved",
        ]

        # Check required columns in daily_issue_df
        if not check_columns(daily_issue, daily_cols):
            raise TypeError(f"Columns are missing. Expected: {daily_cols}")

        daily_issue.loc[:, "created_at"] = pd.to_datetime(
            daily_issue["created_at"], format="%Y-%m-%dT%H:%M:%S.%f"
        )
        daily_issue = daily_issue.sort_values(by=["created_at"])
        daily_issue.reset_index(inplace=True, drop=True)

        dfgrouped_branch_name = daily_issue.groupby("branch_name")
        uniquebranches = daily_issue.branch_name.unique()
        for branch in uniquebranches:
            # Temp subsets issue data basis different branches
            temp = dfgrouped_branch_name.get_group(branch)
            # temp = daily_issue[daily_issue.branch_name == branch]
            temp.reset_index(inplace=True, drop=True)

            temp_unique_baseoids = temp.base_oid.unique()
            # Calculate fix Rate for each base_oid differently -
            # Meaning for each part taken out from master -
            # Each part would have different behaviour
            if temp_unique_baseoids.shape[0] == 1:
                self.perform_daily_analysis(temp, temp["shortcode"][0])
            else:
                for base_oid in temp_unique_baseoids:
                    if base_oid is not None:
                        # Temp_1 subsetted on branch level data basis different base_oids
                        temp_1 = temp[temp.base_oid == base_oid]
                        temp_1.reset_index(inplace=True, drop=True)
                        self.perform_daily_analysis(
                            temp_1, temp["shortcode"][0])

    def apply(self):
        """Perform both historical and daily analysis."""
        # STEP 1 - HISTORICAL ANALYSIS
        if not self.historical_issue_df.empty:
            self.process_historical_data()
        # STEP 2 - DAILY ANALYSIS
        if not self.daily_issue_df.empty:
            self.process_daily_data()

        if len(self.rows_list) == 0:
            raise NoRowsCreatedWhilePreprocessing(
                f"No rows were created during preprocessing for repo_id {self.repo_id}"
            )

        final_df = pd.DataFrame(self.rows_list)

        # Perform Feature creation function
        feature_records = create_features(
            final_df, self.issue_type, self.severity, self.last_fixed_duration
        )

        fix_rate_records = final_df.to_dict("records")
        preprocessed_records = feature_records.to_dict("records")

        return fix_rate_records, preprocessed_records
