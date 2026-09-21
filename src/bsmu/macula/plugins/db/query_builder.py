from PySide6.QtSql import QSqlQuery

from bsmu.macula.plugins.db.database_manager import DatabaseManager


class QueryBuilder:
    def __init__(self, table: str, db: DatabaseManager):
        self.table = table
        self.database_manager = db
        self.query = None

    def _bind(self, data: dict, prefix=":"):
        for k, v in data.items():
            self.query.bindValue(f"{prefix}{k}", v)

    def insert(self, data: dict):
        self.query = QSqlQuery(self.database_manager.get_connection())
        fields = ", ".join(data.keys())
        placeholders = ", ".join([f":{k}" for k in data])
        sql = f"INSERT INTO {self.table} ({fields}) VALUES ({placeholders})"
        self.query.prepare(sql)
        self._bind(data)
        return self.query

    def update(self, data: dict, where: dict):
        self.query = QSqlQuery(self.database_manager.get_connection())
        set_clause = ", ".join([f"{k} = :{k}" for k in data])
        where_clause = " AND ".join([f"{k} = :where_{k}" for k in where])
        sql = f"UPDATE {self.table} SET {set_clause} WHERE {where_clause}"
        self.query.prepare(sql)
        self._bind(data)
        self._bind(where, prefix=":where_")
        return self.query

    def delete(self, where: dict):
        self.query = QSqlQuery(self.database_manager.get_connection())
        where_clause = " AND ".join([f"{k} = :where_{k}" for k in where])
        sql = f"DELETE FROM {self.table} WHERE {where_clause}"
        self.query.prepare(sql)
        self._bind(where, prefix=":where_")
        return self.query

    def select(self, fields: list = None, filters: dict = None):
        self.query = QSqlQuery(self.database_manager.get_connection())
        # Формируем список полей для SELECT
        field_list = ", ".join(fields) if fields else "*"

        # Формируем SQL-запрос
        if filters:
            conditions = " AND ".join([f"{k} = :{k}" for k in filters])
            sql = f"SELECT {field_list} FROM {self.table} WHERE {conditions}"
            self.query.prepare(sql)
            self._bind(filters)
        else:
            sql = f"SELECT {field_list} FROM {self.table}"
            self.query.prepare(sql)

        return self.query
