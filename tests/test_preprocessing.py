#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test for deployflag/core/Preprocessing module."""
import numpy as np
import pandas as pd
import pytest

from deployflag.core.preprocessing import FixRateCalculator, create_features


def test_fixrate_calculator_apply():
    """Check if preprocessing computed is equal to test file."""
    test_df = pd.read_json("tests/testingFiles/test_fr_df_12860.json")
    del test_df["timestamp"]

    issues = ["CRT-A0014", "CRT-A0016", "GSC-G104", "DOK-DL4006"]
    count = 0
    for each_issue in issues:
        name_path1 = "tests/testingFiles/analysis-runs-12860-" + each_issue + ".json"
        name_path2 = "tests/testingFiles/issue-history-12860-" + each_issue + ".json"

        daily_issue_df = pd.read_json(name_path1)
        historical_issue_df = pd.read_json(name_path2)

        fr_calculator = FixRateCalculator(
            daily_issue_df=daily_issue_df,
            historical_issue_df=historical_issue_df,
            repo_id=12860,
            issue_type="Anti-pattern",
            severity=56,
        )

        df = fr_calculator.apply()
        df = pd.DataFrame(df[0])

        del df["timestamp"]

        test_df_tmp = test_df[test_df.issue_id == each_issue]
        test_df_tmp.reset_index(drop=True, inplace=True)

        assert np.allclose(
            df.select_dtypes(exclude=[object]),
            test_df_tmp.select_dtypes(exclude=[object]),
        )
        assert df.select_dtypes(include=[object]).equals(
            test_df_tmp.select_dtypes(include=[object])
        )
        count += 1

    assert count == len(issues)


def test_create_features():
    """Check if features created are similar in test_df."""
    test_df = pd.read_json("tests/testingFiles/test_pp_df_12860_new.json")
    issues = ["CRT-A0014", "DOK-DL4006", "CRT-A0016", "GSC-G104"]
    issue_types = ["anti-pattern", "anti-pattern", "anti-pattern", "bug risk"]
    severity_scores = [56, 78, 34, 49]

    count = 0
    for i, issue in enumerate(issues):
        name_path = "tests/testingFiles/fr_df_12860_" + issue + "_output.json"
        data = pd.read_json(name_path)
        issue_type = issue_types[i]
        severity = severity_scores[i]
        last_fixed_duration = 20

        df = pd.DataFrame(
            create_features(data, issue_type, severity, last_fixed_duration).to_dict(
                "records"
            )
        )

        del df["timestamp"]

        test_df_tmp = test_df[test_df.issue_id == issue]
        test_df_tmp.reset_index(drop=True, inplace=True)

        assert np.allclose(
            df.select_dtypes(exclude=[object]),
            test_df_tmp.select_dtypes(exclude=[object]),
        )
        assert df.select_dtypes(include=[object]).equals(
            test_df_tmp.select_dtypes(include=[object])
        )

        count += 1
    assert count == len(issues)


def test_wrong_cols_daily():
    """Test if TypeError is raise when wrong cols are given in daily_issue_df."""
    daily_issue_df = pd.read_json(
        "tests/testingFiles/wrong-cols-analysis-runs-12860-CRT-A0014.json"
    )
    historical_issue_df = pd.read_json(
        "tests/testingFiles/issue-history-12860-CRT-A0014.json"
    )

    fr_calculator = FixRateCalculator(
        daily_issue_df=daily_issue_df,
        historical_issue_df=historical_issue_df,
        repo_id=12860,
        issue_type="Anti-pattern",
        severity=56,
    )

    with pytest.raises(TypeError):
        fr_calculator.apply()


def test_wrong_cols_hist():
    """Test if TypeError is raise when wrong cols are given in historical_issue_df."""
    daily_issue_df = pd.read_json(
        "tests/testingFiles/analysis-runs-12860-CRT-A0014.json"
    )
    historical_issue_df = pd.read_json(
        "tests/testingFiles/wrong-cols-issue-history-12860-CRT-A0014.json"
    )

    fr_calculator = FixRateCalculator(
        daily_issue_df=daily_issue_df,
        historical_issue_df=historical_issue_df,
        repo_id=12860,
        issue_type="Anti-pattern",
        severity=56,
    )

    with pytest.raises(TypeError):
        fr_calculator.apply()


def test_wrong_cols_create_features():
    """Check if TypeError is raise when creating features with wrong cols."""
    data = pd.read_json(
        "tests/testingFiles/wrong_cols_fr_df_12860_CRT-A0014_output.json"
    )
    issue_type = "Anti-pattern"
    severity = 56
    last_fixed_duration = 20

    with pytest.raises(TypeError):
        create_features(data, issue_type, severity, last_fixed_duration)
