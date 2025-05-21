from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QStyledItemDelegate, QWidget, QHBoxLayout, QPushButton, QMessageBox


class EditPacientDelegate(QStyledItemDelegate):
    def __init__(self, icon_path, parent=None, callback=None):
        super().__init__(parent)
        self.icon_path = icon_path
        self.callback = callback  # Функция, которая вызывается при нажатии кнопки

    def paint(self, painter, option, index):
        """Рисуем кнопку в ячейке"""
        icon = QIcon(self.icon_path)  # 🔹 Здесь нужна иконка (например, карандаш)
        icon.paint(painter, option.rect)

    def editorEvent(self, event, model, option, index):
        """Обрабатываем нажатие на кнопку"""
        if event.type() == event.Type.MouseButtonPress:
            self.callback(model.data(model.index(index.row(), 0)))
        return True