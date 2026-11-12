from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QComboBox


class WheelBlocker(QObject):
    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.Wheel:
            return False
        if not isinstance(obj, QComboBox):
            return False
        # Блокируем прокрутку, если список не открыт
        return not obj.view().isVisible()
