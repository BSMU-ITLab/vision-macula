"""Тесты окна базы данных: соединение, запросы, диалоги и формы.

Работают на копии реальной ``plugins/db/sql/database.db`` в temp-каталоге,
поэтому ничего в репозитории не меняют. Qt — в offscreen-режиме.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSql import QSqlQuery, QSqlQueryModel
from PySide6.QtWidgets import QApplication, QMessageBox, QStyleOptionViewItem

from bsmu.macula.plugins.db import BD as bd_module
from bsmu.macula.plugins.db import constants as db_constants
from bsmu.macula.plugins.db.add_appoitment_dialog import AddAppoitmentRecordDialog
from bsmu.macula.plugins.db.add_patient_dialog import AddRecordDialog
from bsmu.macula.plugins.db.database_manager import DatabaseManager
from bsmu.macula.plugins.db.DynamicFormBuilder import DynamicFormBuilder
from bsmu.macula.plugins.db.edit_pacirnt_delegate import EditPacientDelegate
from bsmu.macula.plugins.db.PatientsModel import PatientsModel
from bsmu.macula.plugins.db.query_builder import QueryBuilder
from bsmu.macula.plugins.db.schema import create_database_file
from bsmu.macula.plugins.db.SQLiteTableViewer import TableWidgetExample, eye_index_by_db_value
from bsmu.macula.plugins.db.work_with_pic import create_scroll_area

PROJECT = Path(__file__).resolve().parent.parent
SOURCE_DB = PROJECT / "bsmu/macula/plugins/db/sql/database.db"

#: Индексы полей из COLUMN_INDEX_TO_FIELD_NAME, которые в таблице eyes
#: называются иначе.
IGNORED_FIELD_INDICES = {5: "topkon", 6: "areds"}


class DbTestCase(unittest.TestCase):
    """Базовая обвязка: копия реальной БД и общее соединение."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        # Уникальный подкаталог: TempDirectory по умолчанию переиспользует
        # освободившееся имя, а на Windows каталог может остаться от процесса
        # с другими правами — тогда копирование файла падает с PermissionError.
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self._tmp_dir = Path(self._tmp.name) / f"db-{uuid4().hex[:8]}"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self._tmp_dir / "database.db"
        shutil.copy(SOURCE_DB, self.db_path)

        self.manager = DatabaseManager()
        self.manager.reset_instance()
        self.addCleanup(DatabaseManager.reset_instance)
        self.db = self.manager.open(self.db_path)

    def scalar(self, sql: str, **params):
        query = QSqlQuery(self.db)
        query.prepare(sql)
        for name, value in params.items():
            query.bindValue(f":{name}", value)
        self.assertTrue(query.exec(), query.lastError().text())
        self.assertTrue(query.next())
        return query.value(0)


class DatabaseManagerTestCase(DbTestCase):
    def test_open_is_idempotent(self) -> None:
        same = self.manager.open(self.db_path)
        self.assertTrue(self.manager.is_open())
        self.assertIs(same, self.db)

    def test_connection_points_to_copied_file(self) -> None:
        self.assertEqual(self.db.databaseName(), self.db_path.absolute().as_posix())

    def test_close_and_reopen(self) -> None:
        self.manager.close()
        self.assertFalse(self.manager.is_open())
        with self.assertRaises(ConnectionError):
            self.manager.get_connection()
        self.manager.open(self.db_path)
        self.assertTrue(self.manager.is_open())

    def test_open_missing_file_raises(self) -> None:
        self.manager.close()
        with self.assertRaises(ConnectionError):
            self.manager.open(self._tmp_dir / "no-such-db.db")


class SchemaConsistencyTestCase(DbTestCase):
    """Код и БД не должны разъезжаться."""

    def _table_columns(self, table: str) -> set[str]:
        query = QSqlQuery(self.db)
        self.assertTrue(query.exec(f"PRAGMA table_info({table})"))
        columns = set()
        while query.next():
            columns.add(query.value(1))
        return columns

    def test_constants_columns_exist_in_eyes(self) -> None:
        columns = self._table_columns(db_constants.EYE_TABLE_NAME)
        self.assertEqual(set(db_constants.COLUMNS_EYES) - columns, set())

    def test_field_name_map_matches_columns(self) -> None:
        columns = self._table_columns(db_constants.EYE_TABLE_NAME)
        for index, field_name in db_constants.COLUMN_INDEX_TO_FIELD_NAME.items():
            if index in IGNORED_FIELD_INDICES:
                continue
            self.assertIn(field_name, columns, f"колонка {field_name} (индекс {index})")

    def test_block_field_indices_are_mapped(self) -> None:
        for block_title, fields in db_constants.BLOCKS:
            for label, field_index in fields:
                self.assertIn(field_index, db_constants.COLUMN_INDEX_TO_FIELD_NAME,
                              f"{block_title}: {label}")

    def test_dropdown_maps_cover_values(self) -> None:
        for field_name, values in db_constants.DROPDOWN_DB_VALUES.items():
            self.assertIn(field_name, db_constants.DROPDOWN_DISPLAY_MAP)
            for value in values:
                self.assertIn(value, db_constants.DROPDOWN_DISPLAY_MAP[field_name])

    def test_injections_table_matches_expected_columns(self) -> None:
        columns = self._table_columns("injections")
        for column in (
            "eye_id", "lutein_therapy", "avastin", "avastin_injections",
            "eylea", "eylea_injections", "visque", "visque_injections",
            "diprospan", "diprospan_injections", "kenalog", "kenalog_injections",
            "lucentis", "lucentis_injections",
        ):
            self.assertIn(column, columns)


class QueryBuilderTestCase(DbTestCase):
    def test_select_uses_real_table(self) -> None:
        builder = QueryBuilder(db_constants.PACIENTS_TABLE_NAME, self.manager)
        query = builder.select(db_constants.PARAMETERS_PATIENTS_SELECT)
        self.assertTrue(query.exec(), query.lastError().text())

        # SQLite не поддерживает size(): считаем строки перебором.
        rows = 0
        while query.next():
            rows += 1
        self.assertGreater(rows, 0)

    def test_insert_creates_patient(self) -> None:
        builder = QueryBuilder(db_constants.PACIENTS_TABLE_NAME, self.manager)
        query = builder.insert({"name": "Тест Тестов", "sex": "М", "year_of_birthday": 1990})
        self.assertTrue(query.exec(), query.lastError().text())

        count = self.scalar("SELECT COUNT(*) FROM pacients WHERE name = :name", name="Тест Тестов")
        self.assertEqual(count, 1)

    def test_delete_builds_where_clause(self) -> None:
        builder = QueryBuilder(db_constants.PACIENTS_TABLE_NAME, self.manager)
        query = builder.delete({"id": 1})
        self.assertIn("DELETE FROM pacients", query.lastQuery())
        self.assertIn("WHERE id = :where_id", query.lastQuery())


class DialogsTestCase(DbTestCase):
    def test_patient_dialog_creates_patient(self) -> None:
        dialog = AddRecordDialog(self.db, 0)
        try:
            dialog.name_input.setText("Диалог Пациентов")
            dialog.sex_input.setText("Ж")
            dialog.age_input.setText("1988")
            dialog.save_data()
        finally:
            dialog.deleteLater()

        count = self.scalar("SELECT COUNT(*) FROM pacients WHERE name = :name",
                            name="Диалог Пациентов")
        self.assertEqual(count, 1)

    def test_patient_dialog_prefills_existing_patient(self) -> None:
        patient_id = self.scalar("SELECT id FROM pacients ORDER BY id LIMIT 1")
        name = self.scalar("SELECT name FROM pacients WHERE id = :id", id=patient_id)

        dialog = AddRecordDialog(self.db, patient_id)
        try:
            self.assertEqual(dialog.name_input.text(), name)
            self.assertFalse(dialog.is_new_user)
        finally:
            dialog.deleteLater()

    def test_patient_dialog_edits_patient(self) -> None:
        """Раньше здесь был AttributeError: вызывался легаси-менеджер."""
        patient_id = self.scalar("SELECT id FROM pacients ORDER BY id LIMIT 1")
        dialog = AddRecordDialog(self.db, patient_id)
        try:
            dialog.name_input.setText("Изменённое Имя")
            dialog.edit_data(patient_id)
        finally:
            dialog.deleteLater()

        self.assertEqual(
            self.scalar("SELECT name FROM pacients WHERE id = :id", id=patient_id),
            "Изменённое Имя",
        )

    def test_patient_dialog_ignores_invalid_input(self) -> None:
        before = self.scalar("SELECT COUNT(*) FROM pacients")
        dialog = AddRecordDialog(self.db, 0)
        try:
            dialog.name_input.setText("")
            dialog.age_input.setText("не-год")
            dialog.save_data()
        finally:
            dialog.deleteLater()
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM pacients"), before)

    def test_appointment_dialog_creates_appointment_and_signals(self) -> None:
        patient_id = self.scalar("SELECT id FROM pacients ORDER BY id LIMIT 1")
        before = self.scalar("SELECT COUNT(*) FROM appointments")

        received: list[dict] = []
        dialog = AddAppoitmentRecordDialog(self.db, patient_id)
        dialog.dataReady.connect(received.append)
        try:
            dialog.duration_input.setText("7")
            dialog.radio_two.setChecked(True)
            dialog.save_data()
        finally:
            dialog.deleteLater()

        self.assertEqual(self.scalar("SELECT COUNT(*) FROM appointments"), before + 1)
        self.assertEqual(len(received), 1)
        self.assertIsNotNone(received[0]["ap_id"])
        self.assertEqual(received[0]["e"], ["L", "R"])

    def test_appointment_dialog_single_eye(self) -> None:
        patient_id = self.scalar("SELECT id FROM pacients ORDER BY id LIMIT 1")
        received: list[dict] = []
        dialog = AddAppoitmentRecordDialog(self.db, patient_id)
        dialog.dataReady.connect(received.append)
        try:
            dialog.duration_input.setText("3")
            dialog.radio_one.setChecked(True)
            dialog.radio_left.setChecked(True)
            dialog.save_data()
        finally:
            dialog.deleteLater()

        self.assertEqual(received[0]["e"], ["L"])

    def test_appointment_dialog_requires_duration(self) -> None:
        patient_id = self.scalar("SELECT id FROM pacients ORDER BY id LIMIT 1")
        before = self.scalar("SELECT COUNT(*) FROM appointments")
        received: list[dict] = []
        dialog = AddAppoitmentRecordDialog(self.db, patient_id)
        dialog.dataReady.connect(received.append)
        try:
            dialog.duration_input.setText("")
            dialog.save_data()
            dialog.close()
        finally:
            dialog.deleteLater()

        self.assertEqual(received, [])
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM appointments"), before)


class EyeIndexTestCase(unittest.TestCase):
    def test_known_values(self) -> None:
        self.assertEqual(eye_index_by_db_value("R"), 0)
        self.assertEqual(eye_index_by_db_value("0"), 0)
        self.assertEqual(eye_index_by_db_value("L"), 1)
        self.assertEqual(eye_index_by_db_value("ос"), 1)

    def test_unknown_value(self) -> None:
        self.assertIsNone(eye_index_by_db_value("??"))


class _Button:
    """Заглушка кнопки для ``toggle_edit_block`` (нужен только текст)."""

    def __init__(self, text: str = "Редактировать"):
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text


class FormBuilderTestCase(DbTestCase):
    """Форма осмотра: виджеты, чтение значений и запись блока в БД."""

    def _build_form(self, row_index: int = 0, eye_index: int = 0):
        """Собирает форму так же, как окно: ключи начальных данных — имена полей."""
        columns = ", ".join(f"eyes.{column}" for column in db_constants.COLUMNS_EYES)
        query = QSqlQuery(self.db)
        self.assertTrue(query.exec(f"SELECT {columns} FROM eyes ORDER BY id LIMIT 5"))
        model = QSqlQueryModel()
        model.setQuery(query)

        initial = {
            field_name: model.index(row_index, column).data()
            for column, field_name in db_constants.COLUMN_INDEX_TO_FIELD_NAME.items()
            if model.index(row_index, column).isValid()
        }
        form = DynamicFormBuilder(
            db_constants.BLOCKS, initial, eye_index, model, self.db, 1, row_index
        )
        self.addCleanup(form.deleteLater)
        return form, model

    def test_form_builds_all_block_fields(self) -> None:
        form, _ = self._build_form()
        expected = len({index for _, fields in db_constants.BLOCKS for _, index in fields})
        self.assertEqual(len(form.get_all_field_widgets()), expected)

    def test_combo_gets_initial_value(self) -> None:
        form, model = self._build_form()
        suffix = DynamicFormBuilder.eye_suffix(0)
        widget = form.get_all_field_widgets().get(f"rpe_status{suffix}")
        self.assertIsNotNone(widget, "поле rpe_status не создано")

        raw_value = model.index(0, 13).data()
        if raw_value is not None and str(raw_value) in db_constants.DROPDOWN_DB_VALUES["rpe_status"]:
            self.assertEqual(str(widget.currentData()), str(raw_value))

    def test_line_edit_gets_initial_value(self) -> None:
        form, model = self._build_form()
        suffix = DynamicFormBuilder.eye_suffix(0)
        widget = form.get_all_field_widgets().get(f"choroidal_thickness_center{suffix}")
        self.assertIsNotNone(widget)

        raw = model.index(0, 9).data()
        expected = "" if raw is None else f"{float(raw):g}" if isinstance(raw, (int, float)) else str(raw)
        self.assertEqual(widget.text(), expected)

    def test_plain_field_keys_fill_widgets(self) -> None:
        """Ключи без суффикса глаза (как их отдаёт окно) должны заполнять поля.

        Раньше форма искала значения по ключу с суффиксом (``rpe_status0``),
        поэтому все поля оставались пустыми.
        """
        model = QSqlQueryModel()
        query = QSqlQuery(self.db)
        self.assertTrue(query.exec(
            "SELECT eyes.eye, eyes.rpe_status, eyes.choroidal_thickness_center "
            "FROM eyes ORDER BY eyes.id LIMIT 1"
        ))
        model.setQuery(query)

        form = DynamicFormBuilder(
            db_constants.BLOCKS,
            {"rpe_status": model.index(0, 1).data(),
             "choroidal_thickness_center": model.index(0, 2).data()},
            0, model, self.db, 1, 0,
        )
        self.addCleanup(form.deleteLater)

        self.assertEqual(
            str(form.get_all_field_widgets()["rpe_status0"].currentData()),
            str(model.index(0, 1).data()),
        )
        self.assertEqual(
            form.get_all_field_widgets()["choroidal_thickness_center0"].text(),
            f"{float(model.index(0, 2).data()):g}",
        )

    def test_empty_data_keeps_fields_empty(self) -> None:
        model = QSqlQueryModel()
        query = QSqlQuery(self.db)
        self.assertTrue(query.exec("SELECT eyes.id FROM eyes ORDER BY eyes.id LIMIT 1"))
        model.setQuery(query)

        form = DynamicFormBuilder(db_constants.BLOCKS, {}, 0, model, self.db, 1, None)
        self.addCleanup(form.deleteLater)

        edit = form.get_all_field_widgets()["choroidal_thickness_center0"]
        self.assertEqual(edit.text(), "")
        # Пустая форма — это режим заполнения нового приёма: поля редактируемы.
        self.assertFalse(edit.isReadOnly())

    def test_save_block_updates_database(self) -> None:
        form, _ = self._build_form(eye_index=0)
        eye_id = form.appointment_data_model.index(0, 0).data()

        combo = form.field_widgets["rpe_status0"]
        target_index = combo.findData("1")
        self.assertGreaterEqual(target_index, 0)
        combo.setCurrentIndex(target_index)

        fields = next(fields for title, fields in db_constants.BLOCKS if title == "")
        form.toggle_edit_block(fields, _Button())
        form.toggle_edit_block(fields, _Button("Сохранить"))

        self.assertEqual(
            str(self.scalar("SELECT rpe_status FROM eyes WHERE id = :id", id=eye_id)), "1"
        )

    def test_clear_all_fields(self) -> None:
        form, _ = self._build_form()
        form.clear_all_fields()
        widget = form.get_all_field_widgets().get("serouz_rpe_detachment_width0")
        if widget is not None and hasattr(widget, "text"):
            self.assertEqual(widget.text(), "")


class WidgetsTestCase(DbTestCase):
    def test_patients_model_adds_edit_column(self) -> None:
        model = PatientsModel()
        query = QSqlQuery(self.db)
        self.assertTrue(query.exec("SELECT id, name, sex, year_of_birthday FROM pacients"))
        model.setQuery(query)

        self.assertEqual(model.columnCount(), 5)
        self.assertEqual(model.data(model.index(0, 4), Qt.DisplayRole), "")
        self.assertIsNotNone(model.data(model.index(0, 1), Qt.DisplayRole))

    def test_create_scroll_area_builds_widget(self) -> None:
        images_dir = self._tmp_dir / "pictures"
        images_dir.mkdir()
        area = create_scroll_area(images_dir)
        try:
            self.assertIsNotNone(area.widget())
        finally:
            area.deleteLater()


class NewDatabaseTestCase(unittest.TestCase):
    """Создание новой базы, когда файла нет."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self._tmp_dir = Path(self._tmp.name) / f"new-{uuid4().hex[:8]}"

    def test_create_database_file_builds_full_schema(self) -> None:
        db_path = self._tmp_dir / "fresh.db"
        create_database_file(db_path)

        self.assertTrue(db_path.exists())
        connection = sqlite3.connect(db_path)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            connection.close()

        # sqlite_sequence появляется вместе с AUTOINCREMENT — это не наша таблица.
        tables.discard("sqlite_sequence")
        self.assertEqual(tables, {"pacients", "appointments", "eyes", "injections"})

    def test_created_schema_matches_real_database(self) -> None:
        """Схема новой базы должна совпадать с рабочей database.db."""
        fresh_path = self._tmp_dir / "fresh.db"
        create_database_file(fresh_path)

        fresh = sqlite3.connect(fresh_path)
        real = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
        try:
            for table in ("pacients", "appointments", "eyes", "injections"):
                fresh_columns = [row[1] for row in fresh.execute(f"PRAGMA table_info({table})")]
                real_columns = [row[1] for row in real.execute(f"PRAGMA table_info({table})")]
                self.assertEqual(fresh_columns, real_columns, f"колонки {table}")

                fresh_types = [row[2] for row in fresh.execute(f"PRAGMA table_info({table})")]
                real_types = [row[2] for row in real.execute(f"PRAGMA table_info({table})")]
                self.assertEqual(fresh_types, real_types, f"типы {table}")
        finally:
            fresh.close()
            real.close()

    def test_created_database_is_usable_by_window(self) -> None:
        """Пустую базу окно должно открыть и показать без пациентов."""
        db_path = self._tmp_dir / "fresh.db"
        create_database_file(db_path)

        window = TableWidgetExample(db_path, self._tmp_dir / "data")
        try:
            self.assertTrue(window.ensure_db_loaded())
            self.assertEqual(window.patients_model.rowCount(), 0)
            self.assertEqual(window.appointments_model.rowCount(), 0)
        finally:
            window.deleteLater()
            DatabaseManager.reset_instance()


class MissingDatabasePromptTestCase(unittest.TestCase):
    """Кнопка Database при отсутствии файла БД."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self._tmp_dir = Path(self._tmp.name) / f"prompt-{uuid4().hex[:8]}"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self._tmp_dir / "database.db"

        self.questions: list[str] = []
        self.critical: list[str] = []
        self.answers: list[QMessageBox.StandardButton] = []

        self._patch_message_box("question", self._fake_question)
        self._patch_message_box("critical", self._fake_critical)

    # --- подмена модальных окон ---

    def _patch_message_box(self, name: str, replacement) -> None:
        patcher = mock.patch.object(bd_module.QMessageBox, name, staticmethod(replacement))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fake_question(self, parent, title, text, *args, **kwargs):
        self.questions.append(text)
        return self.answers.pop(0) if self.answers else QMessageBox.StandardButton.No

    def _fake_critical(self, parent, title, text, *args, **kwargs):
        self.critical.append(text)
        return QMessageBox.StandardButton.Ok

    def _plugin(self):
        """Плагин BD без запуска фреймворка: подменяем только нужные методы."""
        plugin = bd_module.BD.__new__(bd_module.BD)
        plugin._window = None
        plugin._main_window = None
        # Проверяем именно логику «нет файла -> вопрос -> создать/отмена».
        plugin._database_path = lambda: self.db_path
        plugin.config_value = lambda key, default=None: default
        return plugin

    # --- тесты ---

    def test_question_is_asked_and_yes_creates_database(self) -> None:
        plugin = self._plugin()
        self.answers = [QMessageBox.StandardButton.Yes]

        plugin._open_database_window()

        self.assertEqual(len(self.questions), 1)
        self.assertIn(str(self.db_path), self.questions[0])
        self.assertTrue(self.db_path.exists())
        self.assertIsNotNone(plugin._window)

        # Новая база должна открываться и иметь схему.
        connection = sqlite3.connect(self.db_path)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            connection.close()
        self.assertIn("eyes", tables)
        DatabaseManager.reset_instance()

    def test_no_leaves_database_absent_and_window_closed(self) -> None:
        plugin = self._plugin()
        self.answers = [QMessageBox.StandardButton.No]

        plugin._open_database_window()

        self.assertEqual(len(self.questions), 1)
        self.assertFalse(self.db_path.exists())
        self.assertIsNone(plugin._window)
        self.assertEqual(self.critical, [])

    def test_existing_database_opens_without_question(self) -> None:
        shutil.copy(SOURCE_DB, self.db_path)
        plugin = self._plugin()

        plugin._open_database_window()

        self.assertEqual(self.questions, [])
        self.assertIsNotNone(plugin._window)
        DatabaseManager.reset_instance()

    def test_failed_creation_is_reported(self) -> None:
        """Если файл создать нельзя — показываем ошибку и окно не открываем."""
        plugin = self._plugin()
        self.answers = [QMessageBox.StandardButton.Yes]

        with mock.patch.object(
            bd_module, "create_database_file", side_effect=OSError("нет доступа")
        ):
            plugin._open_database_window()

        self.assertTrue(self.critical)
        self.assertIsNone(plugin._window)


class TableWindowTestCase(DbTestCase):
    """Окно БД целиком: пациенты, приёмы и вкладки-формы."""

    def _window(self) -> TableWidgetExample:
        window = TableWidgetExample(self.db_path, self._tmp_dir / "data")
        self.addCleanup(window.deleteLater)
        self.assertTrue(window.ensure_db_loaded())
        return window

    def test_patients_are_loaded(self) -> None:
        window = self._window()
        self.assertGreater(window.patients_model.rowCount(), 0)
        self.assertIsNotNone(window.patients_model.index(0, 1).data())

    def test_click_patient_loads_appointments(self) -> None:
        window = self._window()
        window._on_patient_clicked(window.patients_model.index(0, 0))
        self.assertGreater(window.appointments_model.rowCount(), 0)

    def test_click_appointment_builds_eye_tabs(self) -> None:
        window = self._window()
        window._on_patient_clicked(window.patients_model.index(0, 0))
        window._on_appointment_clicked(window.appointments_model.index(0, 0))

        self.assertGreater(window.appointment_data_model.rowCount(), 0)
        expected_tabs = window.appointment_data_model.rowCount() + 1  # + «Картинки»
        self.assertEqual(window.tab_widget.count(), expected_tabs)
        self.assertTrue(window.forms)

    def test_collect_form_values_reads_widgets(self) -> None:
        window = self._window()
        window._on_patient_clicked(window.patients_model.index(0, 0))
        window._on_appointment_clicked(window.appointments_model.index(0, 0))

        eye_index = next(iter(window.forms))
        values = window._collect_form_values(eye_index)
        self.assertIsInstance(values, dict)
        for field_name in values:
            self.assertIn(field_name, db_constants.COLUMN_INDEX_TO_FIELD_NAME.values())

    def test_form_shows_values_from_database(self) -> None:
        """Данные глаза из БД должны попадать в поля формы (регресс: пустая форма)."""
        # Запросы делаем до создания окна: его закрытие закрывает соединение.
        patient_id = self.scalar("SELECT id FROM pacients ORDER BY id LIMIT 1")
        eye = QSqlQuery(self.db)
        self.assertTrue(eye.exec(
            "SELECT appointment_id, choroidal_thickness_center, rpe_status "
            "FROM eyes WHERE choroidal_thickness_center IS NOT NULL "
            "AND choroidal_thickness_center <> '' AND rpe_status IS NOT NULL "
            "AND rpe_status <> '' LIMIT 1"
        ))
        self.assertTrue(eye.next(), "в базе нет заполненных записей eyes")
        appointment_id, thickness, rpe_status = eye.value(0), eye.value(1), eye.value(2)

        window = self._window()
        window.patient_id_clicked = patient_id
        window.ensure_db_loaded()

        window._load_appointment_data(appointment_id)
        window._rebuild_tabs_for_appointment()

        form = next(iter(window.forms.values()))
        widgets = form.get_all_field_widgets()
        suffix = DynamicFormBuilder.eye_suffix(form.eye_index)

        self.assertEqual(
            widgets[f"choroidal_thickness_center{suffix}"].text(), f"{float(thickness):g}"
        )
        self.assertEqual(
            str(widgets[f"rpe_status{suffix}"].currentData()), str(rpe_status)
        )

    def test_patients_table_headers_are_russian(self) -> None:
        """Заголовки таблицы пациентов должны быть переведены.

        Раньше setHeaderData вызывался из _setup_ui, когда в модели ещё нет
        колонок, и заголовки оставались именами колонок БД.
        """
        window = self._window()
        headers = [
            window.patients_model.headerData(column, Qt.Horizontal)
            for column in range(window.patients_model.columnCount())
        ]
        self.assertEqual(headers, ["Ид", "Имя", "Пол", "Год рождения", "Редактировать"])

    def test_edit_delegate_is_set_on_last_column(self) -> None:
        """Иконка «карандаш» должна стоять в колонке редактирования, а не в -1."""
        window = self._window()
        last_column = window.patients_model.columnCount() - 1
        delegate = window.patients_table.itemDelegateForColumn(last_column)

        self.assertIsInstance(delegate, EditPacientDelegate)
        self.assertFalse(delegate._icon.isNull(), "иконка редактирования не загрузилась")

    def test_edit_delegate_paint_does_not_fail(self) -> None:
        """Отрисовка иконки в ячейку не должна падать."""
        window = self._window()
        delegate = window.patients_table.itemDelegateForColumn(
            window.patients_model.columnCount() - 1
        )
        self.assertIsNotNone(delegate)

        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        option = QStyleOptionViewItem()
        option.rect = QRect(0, 0, 24, 24)
        try:
            delegate.paint(
                painter, option, window.patients_model.index(0, window.patients_model.columnCount() - 1)
            )
        finally:
            painter.end()

    def test_appointments_headers_are_russian(self) -> None:
        window = self._window()
        window._on_patient_clicked(window.patients_model.index(0, 0))

        headers = [
            window.appointments_model.headerData(column, Qt.Horizontal)
            for column in range(window.appointments_model.columnCount())
        ]
        self.assertEqual(headers, ["Ид", "Дата приема", "Длит. болезни"])

    def test_forms_are_rebuilt_between_appointments(self) -> None:
        window = self._window()
        window._on_patient_clicked(window.patients_model.index(0, 0))
        window._on_appointment_clicked(window.appointments_model.index(0, 0))
        first_ids = {id(form) for form in window.forms.values()}

        window._on_appointment_clicked(window.appointments_model.index(0, 0))
        second_ids = {id(form) for form in window.forms.values()}
        self.assertFalse(first_ids & second_ids)

    def test_new_appointment_creates_forms_without_model_rows(self) -> None:
        """Вкладки нового приёма раньше вообще не получали форму (KeyError/None)."""
        patient_id = self.scalar("SELECT id FROM pacients ORDER BY id LIMIT 1")
        window = self._window()
        window.patient_id_clicked = patient_id

        window._on_appointment_created({"e": ["R", "L"], "ap_id": 123456})

        self.assertEqual(window.tab_widget.count(), 3)
        self.assertEqual(sorted(window.forms), [0, 1])
        self.assertTrue(all(form.row_index is None for form in window.forms.values()))


if __name__ == "__main__":
    unittest.main()
