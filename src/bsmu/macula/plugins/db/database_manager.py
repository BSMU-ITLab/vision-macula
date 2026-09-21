"""Единственное соединение с SQLite через QSqlDatabase.

Раньше в пакете было два класса с именем ``DatabaseManager``: этот
(рабочий, на ``QSqlDatabase``) и легаси-файл на ``sqlite3`` со своей схемой
БД, который никто не подключал. Легаси-версия удалена, диалоги переведены
сюда.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtSql import QSqlDatabase

#: Имя соединения по умолчанию (QSqlDatabase требует уникальные имена).
DEFAULT_CONNECTION_NAME = "main_connection"


class DatabaseManager:
    """Ленивое открытие БД и общий доступ к соединению.

    Класс — синглтон: ``TableWidgetExample``, диалоги и формы должны работать
    в одном соединении.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance.db = None
            instance._path = None
            instance._connection_name = DEFAULT_CONNECTION_NAME
            cls._instance = instance
        return cls._instance

    # --- API ---

    def open(self, path: Path | str,
             connection_name: str = DEFAULT_CONNECTION_NAME) -> QSqlDatabase:
        """Открывает (или переоткрывает) БД по указанному пути."""
        candidate = Path(path)
        # SQLite создаёт файл при открытии: без явной проверки опечатка в
        # имени базы давала бы пустое окно вместо понятной ошибки.
        if not candidate.exists():
            raise ConnectionError(f"Файл базы данных не найден: {candidate}")

        path_str = candidate.absolute().as_posix()
        if self.is_open() and self._path == path_str:
            return self.db

        self.close()
        self._connection_name = connection_name
        self._path = path_str

        self.db = QSqlDatabase.addDatabase("QSQLITE", connection_name)
        self.db.setDatabaseName(path_str)
        if not self.db.open():
            error = self.db.lastError().text()
            self.close()
            raise ConnectionError(f"Не удалось открыть базу данных {path_str}: {error}")
        return self.db

    def is_open(self) -> bool:
        return self.db is not None and self.db.isOpen()

    def get_connection(self) -> QSqlDatabase:
        """Активное соединение; исключение, если БД ещё не открыта."""
        if not self.is_open():
            raise ConnectionError("База данных не открыта")
        return self.db

    def close(self) -> None:
        """Закрывает соединение и убирает его из QSqlDatabase."""
        if self.db is not None:
            if self.db.isOpen():
                self.db.close()
            QSqlDatabase.removeDatabase(self._connection_name)
        self.db = None
        self._path = None

    @classmethod
    def reset_instance(cls) -> None:
        """Сбрасывает синглтон (нужно тестам и повторному открытию)."""
        if cls._instance is not None:
            cls._instance.close()
        cls._instance = None
