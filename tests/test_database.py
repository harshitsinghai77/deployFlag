import os
from datetime import datetime

import pytest
from peewee import IntegrityError, PostgresqlDatabase, SqliteDatabase

from deployflag.models.metadata import (
    GridSearchParameter,
    ModelFramework,
    ModelPerformanceMetadata,
)

MODELS = [ModelFramework, GridSearchParameter, ModelPerformanceMetadata]

pg_database = PostgresqlDatabase(
    "deployflag",
    user=os.getenv("PG_DEPLOY_FLAG_DB_NAME", "deployflag"),
    password=os.getenv("PG_DEPLOY_FLAG_DB_PASSWORD", "password"),
    host=os.getenv("PG_DEPLOY_FLAG_DB_HOST", "localhost"),
    port=os.getenv("PG_DEPLOY_FLAG_DB_PORT", "5432"),
)

sqlite_database = SqliteDatabase(
    "dev/sqlite3/test.db",
)


@pytest.mark.parametrize("database", [pg_database, sqlite_database])
def test_database(database):
    """Check for create, read and delete operations in the database."""
    if not database.is_closed():
        database.close()

    # Bind model classes to test db. Since we have a complete list of
    # all models, we do not need to recursively bind dependencies.
    database.bind(MODELS, bind_refs=False, bind_backrefs=False)
    database.connect()

    assert database

    database.create_tables(MODELS)
    assert ModelPerformanceMetadata.table_exists()
    assert GridSearchParameter.table_exists()
    assert ModelFramework.table_exists()

    database.drop_tables(MODELS)
    assert ModelPerformanceMetadata.table_exists() is False
    assert ModelFramework.table_exists() is False
    assert GridSearchParameter.table_exists() is False

    #  Test GridSearchParameter Table
    database.create_tables([GridSearchParameter])
    GridSearchParameter.create(
        version="v1.0",
        objective="binary:logistic",
        eval_metric="logloss",
        parameters={
            "max_depth": [10],
            "n_estimators": [600],
            "learning_rate": [0.01],
            "gamma": [1],
            "colsample_bytree": [0.5],
        },
    )

    dummy_metadata = GridSearchParameter.get(version="v1.0")
    assert dummy_metadata.eval_metric == "logloss"

    with database.atomic():
        # Check if creating a row with same version throws unique constraint IntegrityError.
        with pytest.raises(IntegrityError):
            GridSearchParameter.create(
                version="v1.0",
                objective="binary:logistic",
                eval_metric="logloss",
                parameters={
                    "max_depth": [10],
                    "n_estimators": [600],
                    "learning_rate": [0.01],
                    "gamma": [1],
                    "colsample_bytree": [0.5],
                },
            )

    GridSearchParameter.get(version="v1.0").delete_instance()
    with pytest.raises(Exception):
        GridSearchParameter.get(version="v1.0").delete_instance()

    # Test ModelFramework Table.
    database.create_tables([ModelFramework])
    ModelFramework.create(
        version="v1.0",
        model_name="xgboost.XGBClassifier",
        model_version="1.3.3",
    )

    dummy_metadata = ModelFramework.get(version="v1.0")
    assert dummy_metadata.model_name == "xgboost.XGBClassifier"

    with database.atomic():
        with pytest.raises(IntegrityError):
            ModelFramework.create(
                version="v1.0",
                model_name="xgboost.XGBClassifier",
                model_version="1.3.3",
            )

    ModelFramework.get(version="v1.0").delete_instance()
    with pytest.raises(Exception):
        ModelFramework.get(version="v1.0").delete_instance()

    # Test ModelPerformanceMetadata Table
    database.create_tables(
        MODELS,
        safe=True,
    )

    grid_search = GridSearchParameter.get_or_create(
        version="v1.0",  # unique key
        defaults={
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "parameters": {
                "max_depth": [10],
                "n_estimators": [600],
                "learning_rate": [0.01],
                "gamma": [1],
                "colsample_bytree": [0.5],
            },
        },
    )

    framework = ModelFramework.get_or_create(
        version="v1.0",  # unique key
        defaults={"model_name": "xgboost.XGBClassifier",
                  "model_version": "1.3.3"},
    )

    ModelPerformanceMetadata.create_metadata(
        repo_id=1639,
        precision_score=0.79,
        roc_auc=0.86,
        training_time=250,
        number_of_training_data_rows=100,
        w1=0.3,
        w2=0.3,
        w3=0.2,
        w4=0.2,
        last_training=datetime.now(),
        model_framework=framework[0],
        grid_search=grid_search[0],
        deployflag_version="0.1.0",
    )

    metadata = ModelPerformanceMetadata.get(repo_id=1639)
    assert metadata.repo_id == "1639"
    assert metadata.precision_score == 0.79
    assert metadata.roc_auc == 0.86

    with database.atomic():
        with pytest.raises(IntegrityError):
            ModelPerformanceMetadata.create(
                id=1,
                repo_id=1639,
                precision_score=0.79,
                roc_auc=0.86,
                training_time=250,
                number_of_training_data_rows=100,
                w1_daily_weighted_fix_rate=0.3,
                w2_historical_weighted_fix_rate=0.3,
                w3_fix_probability=0.2,
                w4_severity=0.2,
                last_training=datetime.now(),
                model_framework=10,
                grid_search_model_parameters=10,
                deployflag_version="0.1.0",
            )

    ModelPerformanceMetadata.get(repo_id="1639").delete_instance()
    with pytest.raises(Exception):
        ModelPerformanceMetadata.get(repo_id="1639").delete_instance()

    ModelPerformanceMetadata.delete().execute()
    ModelFramework.delete().execute()
    GridSearchParameter.delete().execute()

    try:
        database.drop_tables(MODELS, safe=True)
    except:
        pass
    database.close()
