"""Плагин «Database»: окно работы с базой пациентов."""

from __future__ import annotations

import sqlite3

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu, MainWindow, MainWindowPlugin

from bsmu.macula.plugins.db.debug_utils import LOGGER_NAME, logger
from bsmu.macula.plugins.db.schema import create_database_file
from bsmu.macula.plugins.db.SQLiteTableViewer import DEFAULT_EDIT_ICON, TableWidgetExample

#: Значения по умолчанию, если в plugins.db.BD.conf.yaml их нет.
DEFAULT_DB_NAME = "database.db"
#: Подкаталог для снимков пациентов (внутри каталога sql).
DATA_DIR_NAME = "data"


class BD(Plugin):
    """Добавляет в меню алгоритмов пункт «Database»."""

    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        "main_window_plugin": "bsmu.vision.plugins.windows.main.MainWindowPlugin",
        "mdi_plugin": "bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin",
    }

    _SQL_DIR_NAME = "sql"
    _DATA_DIRS = (_SQL_DIR_NAME, f"{_SQL_DIR_NAME}/{DATA_DIR_NAME}")

    def __init__(self, main_window_plugin: MainWindowPlugin, mdi_plugin: MdiPlugin):
        super().__init__()
        self._main_window_plugin = main_window_plugin
        self._mdi_plugin = mdi_plugin
        self._main_window: MainWindow | None = None
        self._window: TableWidgetExample | None = None

    def _enable_gui(self) -> None:
        self._main_window = self._main_window_plugin.main_window
        self._main_window.add_menu_action(
            AlgorithmsMenu, self.tr("Database"), self._open_database_window
        )

    def _disable(self) -> None:
        self._window = None
        self._main_window = None

    # --- окно базы ---

    def _database_path(self):
        return self.data_path(self._SQL_DIR_NAME) / self.config_value("db_name", DEFAULT_DB_NAME)

    def _open_database_window(self) -> None:
        """Открывает окно базы; при отсутствии файла предлагает создать его."""
        db_path = self._database_path()

        if not db_path.exists() and not self._offer_to_create_database(db_path):
            logger.info("Открытие базы отменено: файла %s нет", db_path)
            return

        sql_dir = db_path.parent
        self._window = TableWidgetExample(
            db_path,
            sql_dir / DATA_DIR_NAME,
            edit_icon=self.config_value("edit_picture", DEFAULT_EDIT_ICON),
        )
        self._window.setWindowModality(Qt.ApplicationModal)
        self._window.show()

    def _offer_to_create_database(self, db_path) -> bool:
        """Спрашивает пользователя и при согласии создаёт файл базы.

        :return: ``True``, если базу можно открывать.
        """
        answer = QMessageBox.question(
            self._main_window,
            self.tr("База данных не найдена"),
            self.tr(
                "Файл базы данных не найден:\n{path}\n\n"
                "Создать новую базу данных?"
            ).format(path=db_path),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False

        try:
            create_database_file(db_path)
        except (OSError, sqlite3.Error) as exc:
            QMessageBox.critical(
                self._main_window,
                self.tr("Ошибка"),
                self.tr("Не удалось создать базу данных:\n{path}\n\n{error}").format(
                    path=db_path, error=exc
                ),
            )
            logger.error("Не удалось создать базу %s: %s", db_path, exc)
            return False

        logger.info("Создана новая база данных: %s", db_path)
        return True


__all__ = ["BD", "LOGGER_NAME", "logger"]
