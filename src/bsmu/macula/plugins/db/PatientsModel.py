from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtSql import QSqlQueryModel


class PatientsModel(QSqlQueryModel):
    def columnCount(self, parent=QModelIndex()):
        return super().columnCount(parent) + 1  # +1 для кнопки

    def data(self, index, role=Qt.DisplayRole):
        if index.column() == super().columnCount():
            if role == Qt.DisplayRole:
                return ""  # Пусто, но ячейка существует
        return super().data(index, role)