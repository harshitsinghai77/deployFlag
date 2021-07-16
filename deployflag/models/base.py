import datetime

from peewee import DateTimeField, Model

from deployflag.models import DB_CONNECTION


class BaseModel(Model):
    """Base model class that provides essential boilerplate fields and functionality for other models."""

    created_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        """Meta configuration is passed on to subclasses."""

        database = DB_CONNECTION.connection
        legacy_table_names = False  # better table name with "_"
