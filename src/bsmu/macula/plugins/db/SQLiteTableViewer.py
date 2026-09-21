"""Окно базы данных: пациенты, приёмы и формы осмотра.

Модуль только показывает данные и передаёт правки в БД через
:class:`~bsmu.macula.plugins.db.query_builder.QueryBuilder` и ``QSqlQuery``.
Пути и имя файла БД приходят из конфигурации плагина (``plugins.BD.conf.yaml``).
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlQuery, QSqlQueryModel
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

import bsmu.macula.plugins.db.images.dbicons_rc  # noqa: F401  (регистрация иконок)
from bsmu.macula.plugins.db.DynamicFormBuilder import DynamicFormBuilder
from bsmu.macula.plugins.db.PatientsModel import PatientsModel
from bsmu.macula.plugins.db.add_appoitment_dialog import AddAppoitmentRecordDialog
from bsmu.macula.plugins.db.add_patient_dialog import AddRecordDialog
from bsmu.macula.plugins.db.constants import (
    BLOCKS,
    COLUMN_INDEX_TO_FIELD_NAME,
    COLUMNS_EYES,
    DROPDOWN_DB_VALUES,
    DROPDOWN_DISPLAY_MAP,
    ERROR_TEXT,
    EYE_TABLE_NAME,
    PACIENTS_TABLE_NAME,
    PARAMETERS_PATIENTS_SELECT,
)
from bsmu.macula.plugins.db.database_manager import DatabaseManager
from bsmu.macula.plugins.db.debug_utils import log_sql, logger
from bsmu.macula.plugins.db.edit_pacirnt_delegate import EditPacientDelegate
from bsmu.macula.plugins.db.query_builder import QueryBuilder
from bsmu.macula.plugins.db.work_with_pic import create_scroll_area

#: Иконка кнопки «редактировать» (значение по умолчанию для конфига).
DEFAULT_EDIT_ICON = ":/dbicons/edit.png"
#: Заголовки таблиц.
PATIENT_HEADERS = ("Ид", "Имя", "Пол", "Год рождения", "Редактировать")
APPOINTMENT_HEADERS = ("Ид", "Дата приема", "Длит. болезни")
#: Классы колонок eyes, которые подтягиваются вместе с данными глаза.
INJECTION_COLUMNS = (
    "injections.lutein_therapy",
    "injections.avastin",
    "injections.avastin_injections",
    "injections.eylea",
    "injections.eylea_injections",
    "injections.visque",
    "injections.visque_injections",
    "injections.diprospan",
    "injections.diprospan_injections",
    "injections.kenalog",
    "injections.kenalog_injections",
    "injections.lucentis",
    "injections.lucentis_injections",
)
#: Пол глаза в БД -> индекс вкладки (0 — правый, 1 — левый).
EYE_VALUE_TO_INDEX = {"R": 0, "0": 0, "од": 0, "L": 1, "1": 1, "ос": 1}
EYE_TAB_TITLES = ("Правый глаз", "Левый глаз")


def eye_index_by_db_value(eye_value) -> int | None:
    """Индекс вкладки (0/1) по значению колонки ``eye``."""
    return EYE_VALUE_TO_INDEX.get(str(eye_value).strip())


class TableWidgetExample(QWidget):
    """Главное окно работы с базой: список пациентов, приёмов и формы."""

    def __init__(self, bd_path, data_path, edit_icon: str = DEFAULT_EDIT_ICON):
        super().__init__()

        self.base_dir = Path(data_path)
        self.db_path = Path(bd_path)
        self.edit_icon = edit_icon

        self.dropdown_db_values = DROPDOWN_DB_VALUES
        self.dropdown_display_map = DROPDOWN_DISPLAY_MAP

        self.db_manager = DatabaseManager()
        self.db = None
        self._db_loaded = False
        self._db_load_failed = False

        self.query_builder_eye: QueryBuilder | None = None
        self.query_builder_patient: QueryBuilder | None = None

        # Состояние выбора (появляется после кликов по таблицам).
        self.patient_id_clicked: int | None = None
        self.appointment_id: int | None = None
        self.forms: dict[int, DynamicFormBuilder] = {}

        self._setup_ui()

    # ------------------------------------------------------------------ #
    #  ЛЕНИВАЯ ЗАГРУЗКА БД
    # ------------------------------------------------------------------ #

    def ensure_db_loaded(self) -> bool:
        """Открывает БД при первом показе окна."""
        if self._db_loaded:
            return True
        if self._db_load_failed:
            return False

        if not self.db_path.exists():
            self._db_load_failed = True
            QMessageBox.warning(self, "БД не найдена",
                                f"Файл базы данных не найден:\n{self.db_path}")
            return False

        try:
            self.db = self.db_manager.open(self.db_path)
        except ConnectionError as exc:
            self._db_load_failed = True
            QMessageBox.critical(self, ERROR_TEXT, str(exc))
            return False

        self.query_builder_eye = QueryBuilder(EYE_TABLE_NAME, self.db_manager)
        self.query_builder_patient = QueryBuilder(PACIENTS_TABLE_NAME, self.db_manager)

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._db_loaded = True
        self._load_patients_data()
        return True

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.ensure_db_loaded()

    # ------------------------------------------------------------------ #
    #  ИНТЕРФЕЙС
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        self.setWindowTitle("Пациенты")
        self.resize(1200, 800)

        self.main_layout = QHBoxLayout(self)
        self.left_layout = QVBoxLayout()

        self.patients_model = PatientsModel()
        self._setup_patients_table()

        self.appointments_model = QSqlQueryModel(self)
        self.appointment_data_model = QSqlQueryModel(self)
        self._setup_appointments_table()

        self.tab_widget = QTabWidget()

        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addWidget(self.tab_widget)

    def _setup_patients_table(self) -> None:
        self.patients_table = QTableView()
        self.patients_table.setModel(self.patients_model)
        self.patients_table.setSelectionBehavior(QTableView.SelectRows)
        self.patients_table.clicked.connect(self._on_patient_clicked)
        self.patients_table.setColumnHidden(0, True)

        add_button = QPushButton("Добавить пациента")
        add_button.clicked.connect(partial(self._open_patient_dialog, 0))

        self.left_layout.addWidget(self.patients_table)
        self.left_layout.addWidget(add_button)

    def _setup_appointments_table(self) -> None:
        self.appointments_table = QTableView()
        self.appointments_table.setModel(self.appointments_model)
        self.appointments_table.setSelectionBehavior(QTableView.SelectRows)
        self.appointments_table.clicked.connect(self._on_appointment_clicked)

        self.add_button_appointment = QPushButton("Добавить результаты приема")
        self.add_button_appointment.hide()
        self.add_button_appointment.clicked.connect(self._open_appointment_dialog)

        self.left_layout.addWidget(self.appointments_table)
        self.left_layout.addWidget(self.add_button_appointment)

    # ------------------------------------------------------------------ #
    #  ЗАГРУЗКА ДАННЫХ
    # ------------------------------------------------------------------ #

    def _load_patients_data(self) -> None:
        # Без ensure_db_loaded — иначе бесконечная рекурсия.
        patients_query = self.query_builder_patient.select(PARAMETERS_PATIENTS_SELECT)
        patients_query.exec()
        self.patients_model.setQuery(patients_query)
        # Заголовки и делегат — только после setQuery: до него в модели нет
        # колонок, поэтому setHeaderData игнорируется, а делегат попадает на
        # колонку -1 (иконка редактирования не отображается).
        self._apply_patients_decorations()

    def _apply_patients_decorations(self) -> None:
        """Русские заголовки пациентов и иконка редактирования в последней колонке."""
        for column, header in enumerate(PATIENT_HEADERS):
            self.patients_model.setHeaderData(column, Qt.Horizontal, header)

        last_column = self.patients_model.columnCount() - 1
        if last_column < 0:
            return
        self.patients_table.setItemDelegateForColumn(
            last_column,
            EditPacientDelegate(self.edit_icon, self.patients_table, self._open_patient_dialog),
        )

    def _load_appointments(self, patient_id: int) -> None:
        if not self.ensure_db_loaded():
            return

        query = QSqlQuery(self.db)
        query.prepare(
            "SELECT id, date, duration_of_the_disease FROM appointments "
            "WHERE pacient_id = :pacient_id"
        )
        query.bindValue(":pacient_id", patient_id)
        if not query.exec():
            QMessageBox.warning(self, "Ошибка",
                                f"Ошибка загрузки приемов: {query.lastError().text()}")
            return

        self.appointments_model.setQuery(query)
        for column, header in enumerate(APPOINTMENT_HEADERS):
            self.appointments_model.setHeaderData(column, Qt.Horizontal, header)
        self.appointments_table.setColumnHidden(0, True)

    def _load_appointment_data(self, appointment_id: int) -> None:
        if not self.ensure_db_loaded():
            return

        column_clause = ", ".join(
            column if column.startswith("injections.") else f"eyes.{column}"
            for column in (*COLUMNS_EYES, *INJECTION_COLUMNS)
        )
        query_text = (
            f"SELECT {column_clause} FROM eyes "
            "LEFT JOIN injections ON injections.eye_id = eyes.id "
            "WHERE eyes.appointment_id = :appointment_id"
        )

        query = QSqlQuery(self.db_manager.get_connection())
        query.prepare(query_text)
        query.bindValue(":appointment_id", appointment_id)
        if not query.exec():
            QMessageBox.warning(self, "Ошибка", query.lastError().text())
            return

        model = QSqlQueryModel(self)
        model.setQuery(query)
        log_sql(query_text, query, extra={"appointment_id": appointment_id})

        self.appointment_id = appointment_id
        self.appointment_data_model = model

    # ------------------------------------------------------------------ #
    #  ВКЛАДКИ ФОРМ
    # ------------------------------------------------------------------ #

    def _rebuild_tabs_for_appointment(self) -> None:
        """Пересобирает вкладки глаз по загруженной модели приёма."""
        self.tab_widget.clear()
        self.forms.clear()

        for row in range(self.appointment_data_model.rowCount()):
            eye_value = self.appointment_data_model.data(self.appointment_data_model.index(row, 1))
            eye_id = self.appointment_data_model.data(self.appointment_data_model.index(row, 0))
            eye_index = eye_index_by_db_value(eye_value)
            if eye_index is None:
                logger.warning("Неизвестное значение глаза %r в строке %s", eye_value, row)
                eye_index = row
            self.tab_widget.addTab(
                self._create_eye_tab(eye_index, eye_id=eye_id, row_index=row),
                EYE_TAB_TITLES[eye_index] if eye_index < len(EYE_TAB_TITLES) else f"Глаз {eye_index}",
            )

        self.tab_widget.addTab(self._create_images_tab(), "Картинки")

    def _create_eye_tab(self, eye_index: int, eye_id=None, row_index: int | None = None):
        """Вкладка с формой осмотра одного глаза.

        :param eye_id: id записи eyes (``None`` — форма для новой записи);
        :param row_index: строка модели приёма с начальными значениями.
        """
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout()

        initial_data = {}
        if eye_id is not None and row_index is not None:
            initial_data = self._extract_initial_data(self.appointment_data_model, row_index)
            initial_data = {key: value for key, value in initial_data.items() if value}

        form = DynamicFormBuilder(
            BLOCKS,
            initial_data,
            eye_index,
            self.appointment_data_model,
            self.db_manager.get_connection(),
            self.patient_id_clicked,
            row_index,
        )
        self.forms[eye_index] = form
        layout.addWidget(form)

        if eye_id is None:
            save_button = QPushButton("Сохранить")
            save_button.clicked.connect(partial(self._save_new_eye, eye_index))
            layout.addWidget(save_button)

        content.setLayout(layout)
        scroll_area.setWidget(content)
        return scroll_area

    def _create_images_tab(self) -> QScrollArea:
        folder = self.base_dir / str(self.patient_id_clicked) / str(self.appointment_id)
        return create_scroll_area(folder)

    def _extract_initial_data(self, model, row_index: int) -> dict:
        initial_data = {}
        for column, field_name in COLUMN_INDEX_TO_FIELD_NAME.items():
            index = model.index(row_index, column)
            if index.isValid():
                initial_data[field_name] = index.data()
        return initial_data

    # ------------------------------------------------------------------ #
    #  ОБРАБОТЧИКИ
    # ------------------------------------------------------------------ #

    def _on_patient_clicked(self, index) -> None:
        if not self.ensure_db_loaded():
            return

        self.patient_id_clicked = self.patients_model.index(index.row(), 0).data()
        self.add_button_appointment.show()
        self._load_appointments(self.patient_id_clicked)
        self.tab_widget.clear()

    def _on_appointment_clicked(self, index) -> None:
        if not self.ensure_db_loaded():
            return

        appointment_id = self.appointments_model.index(index.row(), 0).data()
        self._load_appointment_data(appointment_id)
        self._rebuild_tabs_for_appointment()

    def _open_patient_dialog(self, patient_id: int) -> None:
        if not self.ensure_db_loaded():
            return

        dialog = AddRecordDialog(self.db, patient_id)
        dialog.exec()
        self._load_patients_data()

    def _open_appointment_dialog(self) -> None:
        if not self.ensure_db_loaded():
            return

        dialog = AddAppoitmentRecordDialog(self.db, self.patient_id_clicked)
        dialog.dataReady.connect(self._on_appointment_created)
        dialog.exec()

    def _on_appointment_created(self, data: dict) -> None:
        """Новый приём создан: показываем по вкладке на каждый выбранный глаз."""
        self.tab_widget.clear()
        self.forms.clear()
        self.appointment_id = data["ap_id"]
        self._create_appointment_folder(self.appointment_id)

        for eye_value in data["e"]:
            eye_index = eye_index_by_db_value(eye_value)
            if eye_index is None:
                continue
            self.tab_widget.addTab(
                self._create_eye_tab(eye_index), EYE_TAB_TITLES[eye_index]
            )

        self.tab_widget.addTab(self._create_images_tab(), "Картинки")

    # ------------------------------------------------------------------ #
    #  СОХРАНЕНИЕ
    # ------------------------------------------------------------------ #

    def _collect_form_values(self, eye_index: int) -> dict[str, object]:
        """Значения заполненных полей формы выбранного глаза."""
        form = self.forms.get(eye_index)
        if form is None:
            return {}

        values: dict[str, object] = {}
        for _, fields in BLOCKS:
            for _, field_index in fields:
                field_name = COLUMN_INDEX_TO_FIELD_NAME.get(field_index)
                widget = form.get_all_field_widgets().get(f"{field_name}{eye_index}")
                if widget is None:
                    continue
                if isinstance(widget, QComboBox):
                    value = widget.currentData()
                elif isinstance(widget, QLineEdit):
                    value = widget.text()
                else:
                    continue
                if value is not None and str(value).strip():
                    values[field_name] = value
        return values

    def _save_new_eye(self, eye_index: int) -> None:
        """Сохраняет результаты приёма для нового глаза."""
        if not self.ensure_db_loaded():
            return

        values = self._collect_form_values(eye_index)
        if not values:
            logger.warning("Форма пустая — сохранять нечего")
            return

        if not self.appointment_id:
            appointment_id = self._create_appointment(values)
            if appointment_id is None:
                return
            self.appointment_id = appointment_id

        values.update({
            "appointment_id": self.appointment_id,
            "eye": eye_index,
            # Оптополь используется по умолчанию (поле в форме не выводится).
            "optopol": True,
        })

        fields = ", ".join(values)
        placeholders = ", ".join(f":{field}" for field in values)
        query_text = f"INSERT INTO {EYE_TABLE_NAME} ({fields}) VALUES ({placeholders})"

        query = QSqlQuery(self.db_manager.get_connection())
        query.prepare(query_text)
        for field, value in values.items():
            query.bindValue(f":{field}", value)
        log_sql(query_text, query, values.keys())

        if not query.exec():
            logger.error("Не удалось добавить запись eyes: %s", query.lastError().text())
            QMessageBox.critical(self, ERROR_TEXT, query.lastError().text())
            return

        logger.info("Добавлена запись eyes для приёма %s", self.appointment_id)
        self._load_appointments(self.patient_id_clicked)

    def _create_appointment(self, values: dict) -> int | None:
        """Приём ещё не создан — создаём его из данных формы."""
        query = QSqlQuery(self.db_manager.get_connection())
        query.prepare(
            "INSERT INTO appointments (pacient_id, date, duration_of_the_disease) "
            "VALUES (:pacient_id, :date, :duration)"
        )
        query.bindValue(":pacient_id", self.patient_id_clicked)
        query.bindValue(":date", values.get("date"))
        query.bindValue(":duration", values.get("duration_of_the_disease"))
        if not query.exec():
            logger.error("Не удалось создать приём: %s", query.lastError().text())
            return None

        appointment_id = query.lastInsertId()
        self._create_appointment_folder(appointment_id)
        logger.info("Создан приём id=%s", appointment_id)
        return appointment_id

    def _create_appointment_folder(self, appointment_id: int) -> None:
        folder = self.base_dir / str(self.patient_id_clicked) / str(appointment_id)
        os.makedirs(folder, exist_ok=True)

    # ------------------------------------------------------------------ #

    def closeEvent(self, event) -> None:
        self.db_manager.close()
        super().closeEvent(event)
