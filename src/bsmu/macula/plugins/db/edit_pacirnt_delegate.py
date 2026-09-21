"""Делегат колонки с иконкой редактирования."""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QStyledItemDelegate, QStyle

# Регистрируем ресурсы иконок здесь же: делегат может импортироваться и
# использоваться без окна БД, а без этой регистрации QIcon(":/dbicons/...")
# возвращает пустую иконку (колонка редактирования выглядит пустой).
from bsmu.macula.plugins.db.images import dbicons_rc  # noqa: F401

#: Колонка с идентификатором записи (скрыта в таблице).
ID_COLUMN = 0
#: Отступ иконки от краёв ячейки (пиксели).
ICON_PADDING = 4


class EditPacientDelegate(QStyledItemDelegate):
    """Рисует иконку в ячейке и вызывает callback по клику."""

    def __init__(self, icon_path: str, parent=None, callback=None):
        super().__init__(parent)
        self.callback = callback
        # Иконка создаётся один раз: paint() вызывается на каждую перерисовку.
        self._icon = QIcon(icon_path)
        if self._icon.isNull():
            # Путь из конфига может быть недоступен — берём иконку из стиля,
            # чтобы колонка редактирования не оставалась пустой.
            self._icon = self._style_icon(parent)

    @staticmethod
    def _style_icon(parent) -> QIcon:
        widget = parent if parent is not None else None
        style = widget.style() if widget is not None else None
        if style is None:
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            style = app.style() if app is not None else None
        if style is None:
            return QIcon()
        return style.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)

    def paint(self, painter, option, index) -> None:
        """Рисует иконку по центру ячейки, не растягивая её на всю ячейку."""
        super().paint(painter, option, index)  # фон и выделение строки
        if self._icon.isNull():
            return

        side = max(1, min(option.rect.width(), option.rect.height()) - 2 * ICON_PADDING)
        icon_rect = QRect(0, 0, side, side)
        icon_rect.moveCenter(option.rect.center())
        self._icon.paint(
            painter, icon_rect, Qt.AlignCenter, QIcon.Mode.Normal, QIcon.State.Off
        )

    def editorEvent(self, event, model, option, index) -> bool:
        if event.type() == event.Type.MouseButtonPress and self.callback is not None:
            self.callback(model.data(model.index(index.row(), ID_COLUMN)))
        return True
