import datetime

from peewee import (
    AutoField,
    CharField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    TextField,
)

from deployflag.models.base import BaseModel


class ModelFramework(BaseModel):
    """Machine learning algorithm metadata used for model training."""

    # unique name to identify the version
    version = CharField(max_length=10, unique=True)
    model_name = CharField(max_length=50)
    model_version = CharField(max_length=10)


class GridSearchParameter(BaseModel):
    """Hyperparamater metadata used for model training."""

    # unique name to identify the version
    version = CharField(max_length=10, unique=True)
    objective = CharField(max_length=50)
    eval_metric = CharField(max_length=50)
    parameters = TextField()


class ModelPerformanceMetadata(BaseModel):
    """Model metadata results after training the model."""

    id = AutoField()

    repo_id = CharField(max_length=50)
    precision_score = FloatField()
    roc_auc = FloatField()
    training_time = FloatField()
    number_of_training_data_rows = IntegerField()

    w1_daily_weighted_fix_rate = FloatField()
    w2_historical_weighted_fix_rate = FloatField()
    w3_fix_probability = FloatField()
    w4_severity = FloatField()

    last_training = DateTimeField(default=datetime.datetime.now())

    model_framework = ForeignKeyField(model=ModelFramework)
    grid_search_model_parameters = ForeignKeyField(model=GridSearchParameter)

    deployflag_version = CharField(max_length=50)

    @classmethod
    def create_metadata(
        cls,
        repo_id: str,
        precision_score: float,
        roc_auc: float,
        training_time: float,
        number_of_training_data_rows: int,
        w1: float,
        w2: float,
        w3: float,
        w4: float,
        last_training,
        model_framework,
        grid_search,
        deployflag_version: str,
    ):
        """Create and save the data in the table."""
        return cls.create(
            repo_id=repo_id,
            precision_score=precision_score,
            roc_auc=roc_auc,
            # time (in minutes) to train the model
            training_time=training_time,
            number_of_training_data_rows=number_of_training_data_rows,
            w1_daily_weighted_fix_rate=w1,
            w2_historical_weighted_fix_rate=w2,
            w3_fix_probability=w3,
            w4_severity=w4,
            last_training=last_training,
            model_framework=model_framework,
            grid_search_model_parameters=grid_search,
            deployflag_version=deployflag_version,
        )
