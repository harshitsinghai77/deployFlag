import os

OCAVE_TASK_NAME = os.getenv(
    "OCAVE_TASK_NAME", "contrib.deployflag.tasks.store_results_from_deployflag"
)
