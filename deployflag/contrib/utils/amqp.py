#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities related to RabbitMQ."""
import atexit

from kombu import Connection
from kombu.pools import producers

from deployflag.logger import LOGGER


class RMQConnection:
    """Create a RMQ connection."""

    def __init__(self, broker_url="amqp://"):
        """Connect to RabbitMQ.

        Args:
            broker_url (string): RMQ connection string.
        """
        self.connection = Connection(broker_url)
        self.connection.connect()

        # register the connection to be closed cleanly when exiting application
        atexit.register(self.connection.release)

    def _revive_connection(self):
        """Try to revive the connection to rabbitmq in case of Timeout or ConnectionResetError."""
        revived_connection = self.connection.clone()
        revived_connection.ensure_connection(max_retries=3)
        revived_connection.connect()
        self.connection = revived_connection

    def publish(self, exchange, routing_key, data, retry=1):
        """
        Publish a message to RabbitMQ.
        Args:
            exchange (string): Name of the exchange to publish to.
            routing_key (string): Routing key to use.
            data (dict): Message which would be serialized to JSON for sending.
        """
        with producers[self.connection].acquire(block=True) as producer:
            try:
                producer.publish(
                    data, exchange=exchange, routing_key=routing_key, serializer="json"
                )
            except (TimeoutError, ConnectionResetError):
                # If it's already been retried then re raise the exception
                if not retry:
                    raise
                self._revive_connection()
                self.publish(exchange, routing_key, data, retry=0)
                LOGGER.info("Retrying publishing message.")

    def __del__(self):
        """Close the connection cleanly when destroying an instance."""
        self.connection.release()
