import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QWidget, QScrollArea, QGridLayout, QDialog, QVBoxLayout


# class ImagePreview(QLabel):
#     def __init__(self, path=None):
#         super().__init__()
#         self.setFixedSize(200, 200)
#         self.setAlignment(Qt.AlignCenter)
#         self.setStyleSheet("border: 1px solid gray;")
#         self.path = path
#         if path:
#             self.setPixmap(QPixmap(path).scaled(200, 200, Qt.KeepAspectRatio))
#
#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton and self.path:
#             self.show_full_image()
#
#     def show_full_image(self):
#         dlg = QDialog()
#         dlg.setWindowTitle("Просмотр изображения")
#         layout = QVBoxLayout(dlg)
#
#         label = QLabel()
#         pixmap = QPixmap(self.path)
#         label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
#         label.setAlignment(Qt.AlignCenter)
#
#         layout.addWidget(label)
#         dlg.resize(820, 620)
#         dlg.exec_()

# class DropWidget(QWidget):
#     def __init__(self, layout):
#         super().__init__()
#         self.layout = layout
#         self.setAcceptDrops(True)
#         self.images = []  # список путей
#
#     def dragEnterEvent(self, event):
#         if event.mimeData().hasUrls():
#             event.acceptProposedAction()
#
#     def dropEvent(self, event):
#         for url in event.mimeData().urls():
#             path = url.toLocalFile()
#             if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
#                 self.images.append(path)
#                 preview = ImagePreview(path)
#                 row, col = divmod(len(self.images) - 1, 3)  # 3 картинки в строке
#                 self.layout.addWidget(preview, row, col)
#         event.acceptProposedAction()


# def create_scroll_area():
#     scroll_area = QScrollArea()
#     scroll_area.setWidgetResizable(True)
#
#     grid_layout = QGridLayout()
#     content = DropWidget(grid_layout)
#
#     # начальные картинки
#     images = [
#         # "C:\\Users\\Evgeniy\\OneDrive\\Pictures\\343433.jpg", "img2.jpg", "img3.jpg"
#     ]
#
#     for i, path in enumerate(images):
#         content.images.append(path)
#         preview = ImagePreview(path)
#         row, col = divmod(i, 3)
#         grid_layout.addWidget(preview, row, col)
#
#     content.setLayout(grid_layout)
#     scroll_area.setWidget(content)
#     return scroll_area
def create_scroll_area(folder_path: str):
    """Создаёт scroll area, загружая картинки из указанной папки"""
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)

    grid_layout = QGridLayout()
    images_dir = Path(folder_path)
    images_dir.mkdir(parents=True, exist_ok=True)

    content = DropWidget(grid_layout, images_dir)

    # Загружаем все картинки из папки
    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))

    for i, path in enumerate(images):
        content.images.append(str(path))
        preview = ImagePreview(str(path))
        row, col = divmod(i, 3)
        grid_layout.addWidget(preview, row, col)

    content.setLayout(grid_layout)
    scroll_area.setWidget(content)
    return scroll_area

class ImagePreview(QLabel):
    """Превью картинки"""
    def __init__(self, path):
        super().__init__()
        self.path = path
        pixmap = QPixmap(path)
        self.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setFixedSize(160, 160)
        self.setAlignment(Qt.AlignCenter)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.show_full_image()

    def show_full_image(self):
        """Открыть картинку в отдельном диалоге"""
        dlg = QDialog()
        dlg.setWindowTitle("Просмотр изображения")
        layout = QVBoxLayout(dlg)

        label = QLabel()
        pixmap = QPixmap(self.path)
        label.setPixmap(pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setAlignment(Qt.AlignCenter)

        layout.addWidget(label)
        dlg.resize(650, 650)
        dlg.exec()


class DropWidget(QWidget):
    """Виджет с поддержкой drag&drop"""
    def __init__(self, grid_layout, images_dir: Path):
        super().__init__()
        self.setAcceptDrops(True)
        self.images = []
        self.layout = grid_layout
        self.images_dir = images_dir

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = Path(url.toLocalFile())
            if file_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                dest = self.images_dir / file_path.name
                shutil.copy(file_path, dest)

                i = len(self.images)
                self.images.append(str(dest))
                preview = ImagePreview(str(dest))
                row, col = divmod(i, 3)
                self.layout.addWidget(preview, row, col)

        event.acceptProposedAction()
