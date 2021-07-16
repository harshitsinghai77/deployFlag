#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Custom exceptions related to deployflag."""


class NoRowsCreatedWhilePreprocessing(Exception):
    """Raised when a no feature rows are created while preprocessing."""
