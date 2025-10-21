from PySide6.QtCore import QObject, QEvent
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QComboBox


class WheelBlocker(QObject):
    def eventFilter(self, obj, event):
        if isinstance(obj, QComboBox) and event.type() == QEvent.Type.Wheel:
            # Блокируем прокрутку, если список не открыт
            if not obj.view().isVisible():
                return True  # игнорируем событие
        return False