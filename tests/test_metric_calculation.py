# -*- coding: utf-8 -*-
"""Test for metric_calculation module."""

import pickle
from math import sqrt

import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

from deployflag.core.metric_calculation import MetricCalculator
from deployflag.core.utils import create_dummy_cols, weighted_average


def test_metric_calculator_transform():
    """Check if margin_error produced and calculated by test_trained_model is ~equal."""
    fr_data = pd.read_json(
        "tests/testingFiles/fr_df_12860_CRT-A0014_output.json")
    preprocessed_data = pd.read_json(
        "tests/testingFiles/pp_df_12860_CRT-A0014_output.json"
    )
    model = pickle.load(open("tests/testingFiles/model_12860.pkl", "rb"))

    metric_model = MetricCalculator(
        fr_data=fr_data,
        preprocessed_data=preprocessed_data,
        repo=12860,
        xg_model=model,
        w1=0.3,
        w2=0.3,
        w3=0.2,
        w4=0.2,
    )

    code_output = pd.DataFrame([metric_model.transform()])

    fr_df = pd.read_csv("tests/testingFiles/test_fr_df_12860.csv")
    fr_df = fr_df[fr_df.issue_id == "CRT-A0014"]
    fr_df.reset_index(drop=True, inplace=True)

    pp_df = pd.read_csv("tests/testingFiles/test_pp_df_12860_new.csv")

    pp_df = create_dummy_cols(
        pp_df,
        [
            "issue_id",
            "quarter",
            "month",
            "freq_detected_quantile",
            "days_after_last_fixed",
            "is_recommended",
            "type",
        ],
    )
    pp_df = pp_df.drop(
        ["repo_id", "fix_rate_binary", "severity_score"],
        axis=1,
    )

    test_model = pickle.load(
        open("tests/testingFiles/test_model_12860.pkl", "rb"))

    rows_list = []

    daily_weighted_fix_rate = weighted_average(
        list(fr_df[fr_df.analysis_type == "daily"]["fix_rate"])
    )

    historical_weighted_fix_rate = weighted_average(
        list(fr_df[fr_df.analysis_type == "historical"]["fix_rate"])
    )

    col_name = "issue_id__" + str("CRT-A0014")
    temp1 = pp_df[pp_df[col_name] == 1]
    pred = test_model.predict_proba(temp1.tail(1))[:, 1]

    severity = 0.56

    final_metric = (
        (0.3 * daily_weighted_fix_rate)
        + (0.3 * historical_weighted_fix_rate)
        + (0.2 * pred[0])
        + (0.2 * severity)
    )

    final_metric = round(final_metric * 100, 0)

    rows_list.append(
        {
            "repo_id": fr_df["repo_id"][0],
            "issue_shortcode": "CRT-A0014",
            "metric": final_metric,
        }
    )

    metric_df = pd.DataFrame(rows_list)

    code_output.sort_values(by=["issue_weight"], inplace=True)
    code_output["issue_weight"] = code_output["issue_weight"].apply(
        lambda x: round(x, 2)
    )
    code_output.reset_index(drop=True, inplace=True)

    metric_df.sort_values(by=["issue_shortcode"], inplace=True)
    metric_df["metric"] = metric_df["metric"].apply(lambda x: round(x, 2))
    metric_df.reset_index(drop=True, inplace=True)

    margin_error = sqrt(
        mean_squared_error(metric_df["metric"], code_output["issue_weight"])
    )

    assert margin_error < 0.015 and code_output.select_dtypes(include=[object]).equals(
        metric_df.select_dtypes(include=[object])
    )


def test_metric_calculator_transform_without_model():
    """Check if metric is calculated without training the model."""
    fr_data = pd.read_json(
        "tests/testingFiles/fr_df_12860_CRT-A0014_output.json")
    preprocessed_data = pd.read_json(
        "tests/testingFiles/pp_df_12860_CRT-A0014_output.json"
    )
    metric_model = MetricCalculator(
        fr_data=fr_data,
        preprocessed_data=preprocessed_data,
        xg_model=None,
        model_store_path="",
        repo=12860,
        w1=0.3,
        w2=0.3,
        w3=0.2,
        w4=0.2,
    )

    code_output = pd.DataFrame([metric_model.transform()])
    assert code_output.loc[0, "issue_weight"] == 17


def test_metric_calculator_transform_with_issue_not_trained():
    """Check metric calculated if issues is not trained in the model."""
    fr_data = pd.read_json(
        "tests/testingFiles/fr_df_12860_ABC-0001_output.json")
    preprocessed_data = pd.read_json(
        "tests/testingFiles/pp_df_12860_ABC-0001_output.json"
    )
    model = pickle.load(open("tests/testingFiles/model_12860.pkl", "rb"))

    metric_model = MetricCalculator(
        fr_data=fr_data,
        preprocessed_data=preprocessed_data,
        xg_model=model,
        repo=12860,
        w1=0.3,
        w2=0.3,
        w3=0.2,
        w4=0.2,
    )

    code_output = pd.DataFrame([metric_model.transform()])
    assert code_output.loc[0, "issue_weight"] == 17


def test_wrong_cols():
    """Check issue raised when wrong cols are passed to MetricCalculator."""
    fr_data = pd.read_json(
        "tests/testingFiles/fr_df_12860_CRT-A0014_output.json")
    preprocessed_data = pd.read_json("tests/testingFiles/wrong_columns.json")
    model = pickle.load(open("tests/testingFiles/model_12860.pkl", "rb"))

    metric_model = MetricCalculator(
        fr_data=fr_data,
        preprocessed_data=preprocessed_data,
        xg_model=model,
        repo=12860,
        w1=0.3,
        w2=0.3,
        w3=0.2,
        w4=0.2,
    )

    with pytest.raises(TypeError):
        metric_model.transform()
