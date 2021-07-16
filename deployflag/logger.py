import logging
import os

import sentry_sdk

sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"))

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger("deployflag")
