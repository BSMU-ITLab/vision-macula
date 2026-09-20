from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QDialog, QVBoxLayout


class ImagePreview(QLabel):
    def __init__(self, img_path, parent=None):
        super().__init__(parent)
        self.img_path = img_path
        pixmap = QPixmap(img_path).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignCenter)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.show_full_image()

    def show_full_image(self):
        dialog = QDialog()
        dialog.setWindowTitle("Полная версия")
        layout = QVBoxLayout(dialog)

        label = QLabel()
        pixmap = QPixmap(self.img_path).scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

        layout.addWidget(label)
        dialog.resize(650, 650)
        dialog.exec()