from PySide6.QtWidgets import (
    QApplication, QWidget, QComboBox, QLabel, QVBoxLayout, QListView
)
from PySide6.QtCore import Qt, QPoint, QEvent, QObject
import sys

class HoverEventFilter(QObject):
    def __init__(self, combo, label):
        super().__init__()
        self.combo = combo
        self.label = label

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseMove:
            index = self.combo.view().indexAt(event.pos())
            if index.isValid():
                tooltip = self.combo.itemData(index.row(), Qt.ItemDataRole.ToolTipRole)
                if tooltip:
                    self.label.setText(tooltip)
                    global_pos = self.combo.view().mapToGlobal(self.combo.view().visualRect(index).topRight())
                    self.label.move(global_pos + QPoint(10, 0))
                    self.label.show()
                else:
                    self.label.hide()
            else:
                self.label.hide()
        elif event.type() in (QEvent.Type.Leave, QEvent.Type.Hide):
            self.label.hide()
        return False

class HoverComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setView(QListView())
        self.popup_label = QLabel("", self)
        self.popup_label.setWindowFlags(Qt.ToolTip)
        self.popup_label.setStyleSheet("background-color: lightyellow; padding: 5px; border: 1px solid gray;")
        self.view().viewport().installEventFilter(HoverEventFilter(self, self.popup_label))

    def addItem(self, text, db_value=None, tooltip=None):
        super().addItem(text, db_value)
        if tooltip:
            index = self.count() - 1
            self.setItemData(index, tooltip, Qt.ItemDataRole.ToolTipRole)


    def hideEvent(self, event):
        self.popup_label.hide()
        super().hideEvent(event)