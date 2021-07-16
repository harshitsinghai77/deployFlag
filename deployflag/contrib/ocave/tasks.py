import warnings
from datetime import datetime

import pandas as pd
from pandas.core.common import SettingWithCopyWarning

from deployflag.celery import RMQ_CONNECTION, celery_app
from deployflag.contrib.ocave.contracts import pack_ocave_message
from deployflag.core import metric_calculation, preprocessing, training
from deployflag.core.constants import (
    DEPLOY_FLAG_VERSION,
    ML_CLASSIFER_NAME,
    ML_CLASSIFIER_EVAL_METRIC,
    ML_CLASSIFIER_OBJECTIVE,
    ML_CLASSIFIER_VERSION,
    ML_GRID_SEARCH_HYPERPARAMETER,
    ML_GRID_SEARCH_PARAMETER_VERSION,
    ML_MODEL_VERSION,
    W1_DAILY_WEIGHTED_FIX_RATE,
    W2_HISTORICAL_WEIGHTED_FIX_RATE,
    W3_FIX_PROBABILITY,
    W4_SEVERITY,
)
from deployflag.core.utils import get_group, is_same
from deployflag.logger import LOGGER
from deployflag.models.metadata import (
    GridSearchParameter,
    ModelFramework,
    ModelPerformanceMetadata,
)

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

severity_score_mapping = {  # To be removed later
    "critical": 82,
    "major": 60,
    "minor": 25,
}


def severity_mapping(severity_type: str) -> int:  # To be removed later
    """
    TODO - This is a temperory approach.
    Retrieve actual weight from concrete_issue__weight in the fruture from the database.
    """
    severity_type = severity_type.lower()
    return severity_score_mapping.get(severity_type, 0)


class Pipeline:
    """Utility class use to initialize deployflag."""

    def __init__(self, repo_id, daily_issue, historical_issue):

        self.repo_id = repo_id
        self.daily_issue = daily_issue
        self.historical_issue = historical_issue

        self.fix_rate_results = {}
        self.preprocess_results = {}
        self.master_preprocessed_result = {}
        self.historical_shortcodes = []
        self.roc_score = 0
        self.precision_score = 0
        self.training_time = 0
        self.xg_model = None
        self.inference_results = []

        # Weights for metric_calculation
        self.w1 = W1_DAILY_WEIGHTED_FIX_RATE
        self.w2 = W2_HISTORICAL_WEIGHTED_FIX_RATE
        self.w3 = W3_FIX_PROBABILITY
        self.w4 = W4_SEVERITY

    def set_historical_shortcodes(self):
        """Get all the shortcodes in historical_issue."""
        self.historical_shortcodes = self.historical_issue["shortcode"].unique(
        )

    def preprocessing(self):
        """
        Looping through all the unique shortcode, grouping and preprocssing it.
        The result is saved in fix_rate_results and preprocess_results
        """
        dfgrouped_daily = self.daily_issue.groupby("shortcode")
        dfgrouped_historical = self.historical_issue.groupby("shortcode")

        for shortcode in self.historical_shortcodes:
            daily_shortcode_tmp = get_group(dfgrouped_daily, shortcode)
            historical_shortcode_tmp = get_group(
                dfgrouped_historical, shortcode)

            selected_df = (
                historical_shortcode_tmp
                if daily_shortcode_tmp.empty
                else daily_shortcode_tmp
            )
            issue_type = selected_df.iloc[0].issue_type.lower()
            severity_score = severity_mapping(selected_df.iloc[0].severity)

            fr_calculator = preprocessing.FixRateCalculator(
                daily_issue_df=daily_shortcode_tmp,
                historical_issue_df=historical_shortcode_tmp,
                repo_id=self.repo_id,
                issue_type=issue_type,
                severity=severity_score,
            )

            fr_df, pp_df = fr_calculator.apply()

            self.fix_rate_results[f"fr_df_{shortcode}_output"] = fr_df
            self.preprocess_results[f"pp_df_{shortcode}_output"] = pp_df

    def combine_and_merge_preprocessed_results(self):
        """
        Loop through all the preprocessed results and create a master preprocessed dataframe for model training.

        Master preprocessed dataframe contains all preprocessed data merged together.
        """
        master_pp_temp = []
        for result in self.preprocess_results.values():
            for row in result:
                master_pp_temp.append(row)

        master_pp_df = pd.DataFrame(master_pp_temp)

        # Memory Optimization: Reduce memory by specifying column types
        # this will downcast the columns automatically to the smallest possible datatype
        # without losing any information.
        floats = master_pp_df.select_dtypes(
            include=["float64"]).columns.tolist()
        master_pp_df[floats] = master_pp_df[floats].apply(
            pd.to_numeric, downcast="float"
        )
        ints = master_pp_df.select_dtypes(include=["int64"]).columns.tolist()
        master_pp_df[ints] = master_pp_df[ints].apply(
            pd.to_numeric, downcast="integer")

        # Removing timestamp as it's not required for model training
        master_pp_df.drop(["timestamp"], axis=1, inplace=True)
        master_pp_df.reset_index(drop=True, inplace=True)
        self.master_preprocessed_result = master_pp_df

    def training(self):
        """
        Traing the model on master preprocessed dataframe.

        Output -> Roc score, Precision score, training time, ml model
        """
        # Check the length of input data - perform training if the data has some rows
        if len(self.master_preprocessed_result) < 1:
            LOGGER.info(
                "Cannot train for repo_id %s, No data found for training.",
                self.repo_id,
            )
            return

        # Check if the y-label only has 1 class. No training can be done if only class is present.
        if is_same(self.master_preprocessed_result["fix_rate_binary"]):
            LOGGER.info(
                "Cannot train for repo_id %s, the least populated class in fix_rate_binary has only 1 members",
                self.repo_id,
            )
            return

        model_trainer = training.ModelTrainer(
            df=self.master_preprocessed_result, repo_id=self.repo_id
        )

        (
            output_roc_score,
            output_precision_score,
            total_training_time,
            xg_clf,
        ) = model_trainer.fit()

        self.roc_score = round(output_roc_score, 2)
        self.precision_score = round(output_precision_score, 2)
        # converting sec to min and rounding off
        self.training_time = round(total_training_time / 60, 2)
        self.xg_model = xg_clf

    def inference(self):
        """
        Output a score btw 1-100 for given issue_id.

        Loop through all the shortcode.
        For each shortcode, generate a score and save the result in an array
        """
        inference_results = []

        if not self.xg_model:
            LOGGER.info("No model found fo this repo %s", self.repo_id)
            LOGGER.info("Setting fix_probab to 0")

        for shortcode in self.historical_shortcodes:
            metric_model = metric_calculation.MetricCalculator(
                fr_data=pd.DataFrame.from_dict(
                    self.fix_rate_results[f"fr_df_{shortcode}_output"]
                ),
                preprocessed_data=pd.DataFrame.from_dict(
                    self.preprocess_results[f"pp_df_{shortcode}_output"]
                ),
                xg_model=self.xg_model,
                repo=self.repo_id,
                w1=self.w1,
                w2=self.w2,
                w3=self.w3,
                w4=self.w4,
            )

            issue_result = metric_model.transform()
            inference_results.append(issue_result)

        self.inference_results = inference_results

    def save_metadata_to_database(self):
        """Save the model metadata to the database for future analysis."""
        # Check if it already exists, if not then create one with the new data inside defaults
        grid_search = GridSearchParameter.get_or_create(
            version=ML_GRID_SEARCH_PARAMETER_VERSION,  # unique key
            defaults={
                "objective": ML_CLASSIFIER_OBJECTIVE,
                "eval_metric": ML_CLASSIFIER_EVAL_METRIC,
                "parameters": ML_GRID_SEARCH_HYPERPARAMETER,
            },
        )

        # Check if it already exists, if not then create one with the new data inside defaults
        framework = ModelFramework.get_or_create(
            version=ML_MODEL_VERSION,  # unique key
            defaults={
                "model_name": ML_CLASSIFER_NAME,
                "model_version": ML_CLASSIFIER_VERSION,
            },
        )

        # creating and saving the model performance metadata
        ModelPerformanceMetadata.create_metadata(
            repo_id=self.repo_id,
            precision_score=self.precision_score,
            roc_auc=self.roc_score,
            training_time=self.training_time,
            number_of_training_data_rows=len(self.master_preprocessed_result),
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
            w4=self.w4,
            last_training=datetime.now(),
            model_framework=framework[0],
            grid_search=grid_search[0],
            deployflag_version=DEPLOY_FLAG_VERSION,
        )

    def send_results_to_ocave(
        self, DEPLOY_FLAG_EXCHANGE_NAME="deployflag", routing_key="deployflag_store_result"
    ):
        """
        Send the result to OCAVE via RMQ to enqueue a run.

        Args:
            data (json): Message data containing repo_id, issue_id, score

        Returns: None.
        """
        ocave_data = pack_ocave_message(
            deployflag_results=self.inference_results, repo_id=self.repo_id
        )

        RMQ_CONNECTION.publish(DEPLOY_FLAG_EXCHANGE_NAME,
                               routing_key, ocave_data)

    def start(self):
        """Run the ml training and inference pipeline."""
        self.set_historical_shortcodes()
        self.preprocessing()
        self.combine_and_merge_preprocessed_results()
        self.training()
        self.inference()
        self.save_metadata_to_database()
        self.send_results_to_ocave()
        LOGGER.info("Results sent to Ocave for repo_id=%s", str(self.repo_id))


@celery_app.task
def process_data_from_ocave(ocave_date):
    """
    Recieve json data from ocave.

    Input -
    ocave_date = JSON Data containing repo_id, default_branch_name, analysis_run
        and "historical_run"
    """
    try:
        repo_id = ocave_date["repo_id"]
        df_daily = pd.DataFrame(ocave_date["analysis_run"])
        df_historical = pd.DataFrame(ocave_date["historical_run"])

        pipeline = Pipeline(
            repo_id=repo_id,
            daily_issue=df_daily,
            historical_issue=df_historical,
        )

        pipeline.start()

    except KeyError as e:
        LOGGER.error(
            "Could not find expected key %s when processing data from ocave_date for repo id %s",
            e,
            str(repo_id),
        )
