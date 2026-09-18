from PySide6.QtSql import QSqlDatabase


class DatabaseManager:
    """Класс для управления соединением с базой данных"""
    _instance = None

    def __new__(cls, path):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls.path = path
            cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        self.db = QSqlDatabase.addDatabase("QSQLITE", "main_connection")
        self.db.setDatabaseName(self.path.absolute().as_posix())
        if not self.db.open():
            raise ConnectionError("Ошибка подключения к базе данных")

    def get_connection(self) -> QSqlDatabase:
        """Возвращает активное соединение с БД"""
        if not self.db.isOpen():
            self._init_db()
        return self.db

    def close(self):
        """Закрывает соединение с БД"""
        if self.db.isOpen():
            self.db.close()
        QSqlDatabase.removeDatabase("main_connection")