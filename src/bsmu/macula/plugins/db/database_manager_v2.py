from PySide6.QtSql import QSqlDatabase


class DatabaseManager:
    """Класс для управления соединением с базой данных (ленивое открытие)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.db = None
            cls._instance._path = None
        return cls._instance

    # --- API ---

    def open(self, path):
        """Открывает (или переоткрывает) БД по указанному пути."""
        path_str = path.absolute().as_posix()

        if self.db is not None and self.db.isOpen() and self._path == path_str:
            return self.db

        self.close()  # закрыть предыдущее, если было

        self._path = path_str
        self.db = QSqlDatabase.addDatabase("QSQLITE", "main_connection")
        self.db.setDatabaseName(path_str)
        if not self.db.open():
            err = self.db.lastError().text() if self.db else "unknown error"
            QSqlDatabase.removeDatabase("main_connection")
            self.db = None
            self._path = None
            raise ConnectionError(f"Ошибка подключения к базе данных: {err}")
        return self.db

    def is_open(self) -> bool:
        return self.db is not None and self.db.isOpen()

    def get_connection(self) -> QSqlDatabase:
        """Возвращает активное соединение с БД."""
        if not self.is_open():
            raise ConnectionError("БД не открыта")
        return self.db

    def close(self):
        """Закрывает соединение с БД."""
        if self.db is not None:
            if self.db.isOpen():
                self.db.close()
            QSqlDatabase.removeDatabase("main_connection")
        self.db = None
        self._path = None