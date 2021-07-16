# -*- coding: utf-8 -*-
"""Test for Training module."""
import pickle

import pandas as pd
from sklearn.metrics import roc_auc_score

from deployflag.core.training import ModelTrainer
from deployflag.core.utils import create_dummy_cols


def test_modeltrainer_fit():
    """Check for model training."""
    test_data = pd.read_csv("tests/testingFiles/test_pp_df_12860_new.csv")

    training_data = pd.read_json("tests/testingFiles/pp_df_12860_output.json")
    model = ModelTrainer(
        df=training_data,
        repo_id=12860,
        save_model=True,
        model_path="tests/testingFiles",
    )

    roc_score, _, _, _ = model.fit()

    test_data = create_dummy_cols(
        test_data,
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
    df_y = test_data[["fix_rate_binary"]]
    test_data = test_data.drop(
        ["repo_id", "fix_rate_binary", "severity_score"],
        axis=1,
    )

    test_model = pickle.load(
        open("tests/testingFiles/test_model_12860.pkl", "rb"))

    out = test_model.predict(test_data)
    test_roc = roc_auc_score(df_y, out)

    # ROC should be within 4-5% tolerance
    assert abs((test_roc - roc_score) / test_roc) * 100 <= 4
