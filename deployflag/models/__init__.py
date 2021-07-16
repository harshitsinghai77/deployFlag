import os

from peewee import (
    DatabaseProxy,
    PostgresqlDatabase,
    SqliteDatabase,
)

USE_DB = os.getenv("USE_DB", "sqlite3")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "dev/sqlite3/deployflag.db")

if USE_DB == "postgresql":
    database = PostgresqlDatabase(
        "deployflag",
        user=os.getenv("PG_DEPLOY_FLAG_DB_NAME", "deployflag"),
        password=os.getenv("PG_DEPLOY_FLAG_DB_PASSWORD", "password"),
        host=os.getenv("PG_DEPLOY_FLAG_DB_HOST", "localhost"),
        port=os.getenv("PG_DEPLOY_FLAG_DB_PORT", "5432"),
    )
    database.set_time_zone("UTC")
else:
    database = SqliteDatabase(
        SQLITE_DB_PATH,
        pragmas={"journal_mode": "wal", "cache_size": -1024 * 64},
    )


class DatabaseConection:
    """Intialize the database."""

    def __init__(self, db):
        self.connection = DatabaseProxy()
        self.connection.initialize(db)

        self.initialize()

    def initialize(self):
        """Close the connection if already open and create a new connection."""
        if not self.connection.is_closed():
            self.connection.close()

        self.connection.connect()

    def create_table(self):
        """Connect and create tables in the database."""
        from deployflag.models.metadata import (
            GridSearchParameter,
            ModelFramework,
            ModelPerformanceMetadata,
        )

        with self.connection:
            self.connection.create_tables(
                [ModelPerformanceMetadata, GridSearchParameter, ModelFramework],
                safe=True,
            )


DB_CONNECTION = DatabaseConection(database)
DB_CONNECTION.create_table()
