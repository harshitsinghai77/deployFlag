import os

import environ

environ.Env.read_env(env_file=os.getenv("ENVFILEPATH", ".env"))
