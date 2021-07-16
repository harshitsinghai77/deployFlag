import os

from celery import Celery
from kombu import Exchange, Queue

from deployflag.contrib.utils.amqp import RMQConnection

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = int(os.getenv("REDIS_DB", "1"))
IS_RMQ_CONNECTION = int(os.getenv("RMQ_CONNETION", "0"))

if IS_RMQ_CONNECTION:
    RMQ_CONNECTION = RMQConnection(CELERY_BROKER_URL)
else:
    RMQ_CONNECTION = None


class Config:
    """Celery Config."""

    DEBUG = False

    broker_url = CELERY_BROKER_URL
    result_backend = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

    task_serializer = "json"
    result_serializer = "json"
    accept_content = ["json"]
    include = ["deployflag.contrib.ocave.tasks"]
    timezone = "UTC"
    enable_utc = True
    imports = ["deployflag.contrib.ocave.tasks"]

    task_routes = {
        "deployflag.contrib.ocave.tasks.process_data_from_ocave": {
            "queue": "deployflag_training",
            "routing_key": "deployflag_training",
        }
    }

    task_queues = {
        Queue(
            "deployflag_training",
            exchange=Exchange("deployflag", type="direct"),
            routing_key="deployflag_training",
        )
    }

    task_create_missing_queues = False
    task_track_started = True
    task_acks_late = True
    worker_prefetch_multiplier = 1
    worker_max_tasks_per_child = 1
    worker_proc_alive_timeout = 300
    broker_pool_limit = None
    timezone = "UTC"
    enable_utc = True


class DevelopmentConfig(Config):
    """Celery Config for Development."""

    DEBUG = False


class ProductionConfig(Config):
    """Celery Config for Production."""


# return active config
available_configs = dict(development=DevelopmentConfig,
                         production=ProductionConfig)
selected_config = os.getenv("ENV", "production")
config = available_configs.get(selected_config, "production")


celery_app = Celery("deployflag")
celery_app.config_from_object(config)

if __name__ == "__main__":
    celery_app.start()
