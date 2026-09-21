"""Модель таблицы пациентов: лишняя колонка под иконку редактирования."""

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtSql import QSqlQueryModel

#: Колонка с идентификатором записи (скрыта в таблице).
ID_COLUMN = 0


class PatientsModel(QSqlQueryModel):
    """Добавляет к выборке пустую колонку для делегата с иконкой."""

    def columnCount(self, parent=QModelIndex()) -> int:
        return super().columnCount(parent) + 1

    def data(self, index, role=Qt.DisplayRole):
        if index.column() == super().columnCount() and role == Qt.DisplayRole:
            return ""
        return super().data(index, role)
