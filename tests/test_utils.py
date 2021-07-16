#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for utils module."""

import numpy as np
import pandas as pd
import pytest

from deployflag.core import utils
from deployflag.core.constants import (
    FREQ_DETECTED_QUANTILE_HIGH,
    FREQ_DETECTED_QUANTILE_LOW,
    FREQ_DETECTED_QUANTILE_MEDIUM_HIGH,
    FREQ_DETECTED_QUANTILE_MEDIUM_LOW,
)


def test_is_same():
    """Check if dataframe contains same elements in given column."""
    assert utils.is_same(pd.Series([1, 1, 1, 1, 1, 1, 1]))


def test_if_all_zero():
    """Check if dataframe contains all zero for a given column."""
    assert utils.if_all_zero(pd.Series([0, 0, 0, 0, 0, 0]))


def test_weighted_average():
    """Check for the weighted average."""
    assert utils.weighted_average([0, 1, 2, 3, 4, 5]) == 3.3333


def test_weighted_average_if_len_zero():
    """Check if weighted_average is zero."""
    assert utils.weighted_average([]) == 0


def test_load_json():
    """Check if json is loaded successfully."""
    df = utils.load_json_as_dataframe(
        "./tests/testingFiles/fr_df_12860_output.json")
    df = df[df.issue_id == "CRT-A0014"]
    df.reset_index(drop=True, inplace=True)
    test_df = pd.read_csv("./tests/testingFiles/test_fr_df_12860.csv")
    test_df = test_df[test_df.issue_id == "CRT-A0014"]
    test_df.reset_index(drop=True, inplace=True)
    del df["timestamp"]
    del test_df["timestamp"]
    assert np.allclose(
        df.select_dtypes(exclude=[object]), test_df.select_dtypes(
            exclude=[object])
    ) & df.select_dtypes(include=[object]).equals(
        test_df.select_dtypes(include=[object])
    )


def test_load_json_raises():
    """Check FileNotFoundError is rasied when file does not exist."""
    with pytest.raises(FileNotFoundError) as execinfo:
        utils.load_json_as_dataframe("")

    assert str(execinfo.value) == "[Errno 2] No such file or directory: ''"


def test_check_columns():
    """Check if given list of column exists in the dataframe."""
    df = utils.load_json_as_dataframe(
        "./tests/testingFiles/fr_df_12860_output.json")
    req_columns = [
        "repo_id",
        "issue_id",
        "timestamp",
        "analysis_type",
        "frequency_detected",
        "fix_rate",
    ]
    columns_not_exist = ["abc", "def"]
    assert utils.check_columns(df, req_columns)
    assert not utils.check_columns(df, columns_not_exist)


def test_create_dummy_cols():
    """Convert categorical variable into dummy/indicator variables."""
    data = {"Col1": [1, 2, 3, 4], "Col2": ["A", "B", "A", "B"]}
    df = pd.DataFrame(data, columns=["Col1", "Col2"])

    df = utils.create_dummy_cols(df, ["Col2"])
    df["Col2__A"] = df["Col2__A"].apply(int)
    df["Col2__B"] = df["Col2__B"].apply(int)

    data = {"Col1": [1, 2, 3, 4], "Col2__A": [
        1, 0, 1, 0], "Col2__B": [0, 1, 0, 1]}

    test_df = pd.DataFrame(data, columns=["Col1", "Col2__A", "Col2__B"])

    assert df.equals(test_df)


def test_calculate_quantiles():
    """Check if values are correct for different quantiles."""
    arr = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    assert utils.calculate_quantiles(arr, 1) == FREQ_DETECTED_QUANTILE_LOW
    assert utils.calculate_quantiles(
        arr, 3) == FREQ_DETECTED_QUANTILE_MEDIUM_LOW
    assert utils.calculate_quantiles(
        arr, 6) == FREQ_DETECTED_QUANTILE_MEDIUM_HIGH
    assert utils.calculate_quantiles(arr, 10) == FREQ_DETECTED_QUANTILE_HIGH


def test_get_group():
    """Check if get_group returns emptry dataframe when shortcode is missing in the dataframe."""
    df = pd.DataFrame(
        {
            "shortcode": [
                "PYL-R1710",
                "PYL-W0107",
                "PYL-W0212",
                "PYL-W0221",
                "PYL-W0223",
                "PYL-W0511",
            ]
        }
    )
    dfgrouped = df.groupby("shortcode")
    assert not utils.get_group(dfgrouped, "PYL-R1710").empty
    assert not utils.get_group(dfgrouped, "PYL-W0221").empty
    assert not utils.get_group(dfgrouped, "PYL-W0511").empty
    assert utils.get_group(dfgrouped, "RANDOM-SHORTCODE").empty
