![deployFlag](images/deployFlag.png?raw=true "Title")

# deployFlag

deployFlag is boilercode for training machine learning algorithms across threads or machines using Celery's distributed task queue.

## Steps to initialize deployFlag

Rename `.env.example` to `.env` or run

```shell
cp .env.example .env
```

```
pyenv virtualenv deployflag-venv
```

Create a file named `.python-version` with content `deployflag-venv` that way your virtual environment will automatically get activated when you `cd` into `deployflag` directory. If for some reason your auto-activation is not working then you can manually activate by `pyenv activate deployflag-venv`.

```
pyenv activate deployflag-venv

make installdeps
    or
poetry install
```

Running the Celery worker

```
CELERY_CONCURRENCY=1 make servedeployFlag
```

or

```
celery -A app worker -l INFO -O fair -Q deployflag_training --without-mingle --without-gossip --hostname deployflagworker@%h

```

## Ocave deployflag Consumer - Producer

Ocave and deployFlag are distributed services which communicate on a producer consumer system pattern. They both run on seperate machines and interact with each other via a broker service (RabbitMQ).

The major components needed to interact between deployFlag and Ocave are:

1. One Exchange in Rabbitmq named 'deployFlag'
2. Queues - 3 (2 for Ocave and 1 for deployFlag)

## Queues and their purpose

1. `deployflag_collect_data` (used and called by Ocave): Responsible for collecting the data from database, packaging it and publishing it in `deployflag_training` queue.

2. `deployflag_training` (used by deployFlag): Receive the data, and call the pipeline which will run various steps related to training the model. Finally the result are sent back to Ocave by publishing it in `deployflag_save_result` queue.

3. `deployflag_save_result` (used by Ocave): Responsible for saving the results in the application database.

## Tests

```shel
make test
```

## Metadata Database

deployFlag uses a database to store model metadata for future analysis. For dev and local setup, SQLite3 will be used and PostgreSQL for production.

Peewee ORM is used for the database.

To change this, simply set `USE_DB=postgresql` or `USE_DB=sqlite3` in .env file.

## SQLite3

The SQLite3 db can be found in /dev/sqlite3/deployflag.db

### Query SQLite for Debugging Locally

Locally query into SQLite3. After activating the `deployflag-venv` type python and enter the shell

```python
from peewee import SqliteDatabase

database = SqliteDatabase(
    "dev/sqlite3/deployflag.db"
)

cursor = database.cursor()

cursor.execute("SELECT * from grid_search_parameter;").fetchall()

cursor.execute("SELECT * from model_performance_metadata;").fetchall()

```

Note: This is just for debugging. Should not be used in production.

## Google Cloud Build

Google Cloud Build will be triggered to automatically build containers from the Dockerfile on every commit to the pull request. Edit `cloudbuild_depl_dev.yaml` for development and `cloudbuild_depl.yaml` for production if required.

## Using Docker container

```
docker build . -t deployflag
```

Make the following changes in the .env file when running Docker container in local.
This is because localhost is not accessible from inside the Docker container.

```
CELERY_BROKER_URL=amqp://guest:guest@<LOCAL-IP>:5672//
REDIS_HOST=<LOCAL-IP>
```

For eg:

```
CELERY_BROKER_URL=amqp://guest:guest@192.168.29.63:5672//
REDIS_HOST=192.168.29.63
```

Run the image

```
docker container run -d <IMAGE_ID> make servedeployFlag
```
