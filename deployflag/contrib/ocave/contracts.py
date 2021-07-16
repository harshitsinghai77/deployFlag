import uuid

from deployflag.contrib.ocave.constants import OCAVE_TASK_NAME


def pack_ocave_message(**kwargs):
    """Packages the data for the celery worker to execute it."""
    return {
        "id": uuid.uuid4(),
        "task": OCAVE_TASK_NAME,  # celery task to trigger in ocave for saving the results
        "kwargs": kwargs,
        "retries": 3,
    }
