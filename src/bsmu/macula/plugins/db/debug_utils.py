"""Логирование SQL-запросов вместо печати в консоль.

Раньше отладочный вывод шёл через ``print`` (в том числе полный дамп таблиц
при каждом клике). Теперь это обычный ``logging`` уровня DEBUG: включается
через ``logging.getLogger('bsmu.macula.plugins.db').setLevel(logging.DEBUG)``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

LOGGER_NAME = "bsmu.macula.plugins.db"

logger = logging.getLogger(LOGGER_NAME)


def log_sql(query_text: str, query=None, fields: Iterable[str] = (),
            extra: dict[str, Any] | None = None) -> None:
    """Пишет текст запроса и связанные значения на уровне DEBUG."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug("SQL:\n%s", query_text.strip())
    for field in fields:
        bound = query.boundValue(f":{field}") if query is not None else None
        logger.debug("  %s = %r", field, bound)
    for name, value in (extra or {}).items():
        logger.debug("  %s = %r", name, value)


#: Прежнее имя функции (использовалось в SQLiteTableViewer).
print_sql_debug = log_sql
