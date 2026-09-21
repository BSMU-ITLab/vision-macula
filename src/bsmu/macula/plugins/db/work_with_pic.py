"""Вкладка «Картинки»: превью изображений приёма и drag&drop файлов."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

#: Колонок превью в сетке.
GRID_COLUMNS = 3
#: Размер превью и полного изображения.
THUMBNAIL_SIZE = 150
PREVIEW_SIZE = 600
#: Поддерживаемые расширения.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


class ImagePreview(QLabel):
    """Превью изображения; по клику открывается в отдельном диалоге."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = str(path)
        pixmap = QPixmap(self.path)
        self.setPixmap(
            pixmap.scaled(THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.setFixedSize(THUMBNAIL_SIZE + 10, THUMBNAIL_SIZE + 10)
        self.setAlignment(Qt.AlignCenter)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.show_full_image()

    def show_full_image(self) -> None:
        dialog = QDialog()
        dialog.setWindowTitle("Просмотр изображения")
        layout = QVBoxLayout(dialog)

        label = QLabel()
        label.setPixmap(
            QPixmap(self.path).scaled(
                PREVIEW_SIZE, PREVIEW_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        dialog.resize(PREVIEW_SIZE + 50, PREVIEW_SIZE + 50)
        dialog.exec()


class DropWidget(QWidget):
    """Сетка превью с поддержкой перетаскивания файлов."""

    def __init__(self, grid_layout: QGridLayout, images_dir: Path):
        super().__init__()
        self.setAcceptDrops(True)
        self.setLayout(grid_layout)
        self.images: list[str] = []
        self.images_dir = images_dir

    def add_image(self, path: str | Path) -> None:
        """Добавляет превью в сетку."""
        index = len(self.images)
        self.images.append(str(path))
        row, column = divmod(index, GRID_COLUMNS)
        self.layout().addWidget(ImagePreview(path), row, column)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            source = Path(url.toLocalFile())
            if source.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            destination = self.images_dir / source.name
            shutil.copy(source, destination)
            self.add_image(destination)
        event.acceptProposedAction()


def create_scroll_area(folder_path: str | Path) -> QScrollArea:
    """Прокручиваемая область с картинками из папки приёма."""
    images_dir = Path(folder_path)
    images_dir.mkdir(parents=True, exist_ok=True)

    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)

    grid_layout = QGridLayout()
    content = DropWidget(grid_layout, images_dir)
    for path in sorted(
        path
        for suffix in IMAGE_SUFFIXES
        for path in images_dir.glob(f"*{suffix}")
    ):
        content.add_image(path)

    scroll_area.setWidget(content)
    return scroll_area
