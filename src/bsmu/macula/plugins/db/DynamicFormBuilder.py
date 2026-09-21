"""Форма осмотра: блоки полей из ``constants.BLOCKS``.

Каждое поле — либо выпадающий список (если имя есть в ``DROPDOWN_DB_VALUES``),
либо строка ввода. По умолчанию поля только для чтения: чтобы править,
нужно нажать «Редактировать» в блоке, тогда кнопка меняется на «Сохранить»
и по нажатию блок уходит в БД.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlQuery
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from bsmu.macula.plugins.db.constants import (
    COLUMN_INDEX_TO_FIELD_NAME,
    DROPDOWN_DB_VALUES,
    DROPDOWN_DISPLAY_MAP,
    DROPDOWN_DISPLAY_MAP2,
)
from bsmu.macula.plugins.db.debug_utils import log_sql, logger
from bsmu.macula.plugins.db.HoverComboBox import HoverComboBox
from bsmu.macula.plugins.db.WheelBlocker import WheelBlocker

EYE_TABLE_NAME = "eyes"
READ_ONLY_STYLE = "background-color: #f0f0f0;"
EDITABLE_STYLE = "background-color: #ffffff;"


class DynamicFormBuilder(QWidget):
    """Форма осмотра одного глаза, собранная из блоков полей."""

    def __init__(self, blocks, initial_data, eye_index, appointment_data_model,
                 db_connection, patient_id_clicked, row_index=None, parent=None):
        super().__init__(parent)
        self.eye_index = eye_index
        self.row_index = row_index
        self.appointment_data_model = appointment_data_model
        self.db_connection = db_connection
        self.patient_id_clicked = patient_id_clicked
        self.initial_data = initial_data or {}
        self.field_widgets: dict[str, QWidget] = {}

        self.main_layout = QVBoxLayout(self)
        self._build_blocks(blocks, eye_index)

    # --- построение ---

    def _build_blocks(self, blocks, eye_index) -> None:
        # Ключи начальных данных — имена полей БД (date, rpe_status, ...):
        # именно так их отдаёт SQLiteTableViewer. Глаза различаются тем, какой
        # строке модели соответствует форма, поэтому суффикс глаза здесь не нужен.
        has_initial_values = self._has_block_values(blocks)

        for block_title, fields in blocks:
            container = QGroupBox(block_title) if block_title else QWidget()
            form_layout = QFormLayout(container)

            for label, field_index in fields:
                field_name = COLUMN_INDEX_TO_FIELD_NAME.get(field_index)
                widget_key = self.widget_key(field_name)
                widget = self._create_field_widget(
                    field_name, self.initial_value(field_name), has_initial_values
                )
                form_layout.addRow(label, widget)
                self.field_widgets[widget_key] = widget

            self.main_layout.addWidget(container)

            if has_initial_values:
                edit_button = QPushButton("Редактировать")
                edit_button.clicked.connect(
                    lambda _, block_fields=fields, button=edit_button:
                    self.toggle_edit_block(block_fields, button)
                )
                form_layout.addRow("", edit_button)

    def widget_key(self, field_name: str) -> str:
        """Ключ виджета в форме: имя поля + суффикс глаза."""
        return f"{field_name}{self.eye_suffix(self.eye_index)}"

    def initial_value(self, field_name: str):
        """Начальное значение поля из модели приёма.

        Понимает оба варианта ключей: ``rpe_status`` (как передаёт окно) и
        ``rpe_status0`` (для совместимости с прежними вызовами).
        """
        if field_name in self.initial_data:
            return self.initial_data[field_name]
        return self.initial_data.get(self.widget_key(field_name))

    def _has_block_values(self, blocks) -> bool:
        """Есть ли данные для полей формы: только тогда блок можно править.

        Смотрим значения именно тех полей, что выводятся в блоках: поле
        ``date`` брать нельзя — оно не выводится и его значение совпадает у
        обоих глаз, из-за чего у второго глаза форма оставалась бы только для
        чтения.
        """
        for _, fields in blocks:
            for _, field_index in fields:
                value = self.initial_value(COLUMN_INDEX_TO_FIELD_NAME.get(field_index))
                if value is not None and str(value).strip():
                    return True
        return False

    @staticmethod
    def eye_suffix(eye_index) -> str:
        """Суффикс имени виджета: номер глаза или ``_`` для новой записи."""
        return str(eye_index if eye_index is not None else "_")

    def _create_field_widget(self, field_name: str, initial_value=None,
                             read_only: bool = True) -> QWidget:
        if field_name in DROPDOWN_DB_VALUES:
            return self._create_combo(field_name, initial_value, read_only)
        return self._create_line_edit(initial_value, read_only)

    def _create_combo(self, field_name: str, initial_value, read_only: bool) -> HoverComboBox:
        combo = HoverComboBox()
        combo.installEventFilter(WheelBlocker(combo))
        combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        display_map = DROPDOWN_DISPLAY_MAP[field_name]
        tooltip_map = DROPDOWN_DISPLAY_MAP2.get(field_name)
        for db_value in DROPDOWN_DB_VALUES[field_name]:
            display_text = display_map.get(db_value, db_value)
            tooltip = (tooltip_map or {}).get(db_value, "")
            if tooltip and tooltip.strip():
                combo.addItem(display_text, db_value, tooltip)
            else:
                combo.addItem(display_text, db_value)

        value = self._as_string(initial_value)
        db_values = DROPDOWN_DB_VALUES[field_name]
        display_values = [self._as_string(item) for item in db_values]
        # Часть списков хранит не строки (например, topkon: [False, True]),
        # поэтому сравниваем и строковые представления значений.
        if value in display_values:
            combo.setCurrentIndex(display_values.index(value))
            if read_only:
                combo.setEnabled(False)
                combo.setStyleSheet(READ_ONLY_STYLE)
        return combo

    @staticmethod
    def _create_line_edit(initial_value, read_only: bool) -> QLineEdit:
        line_edit = QLineEdit()
        if initial_value is not None:
            line_edit.setText(str(initial_value))
        if read_only:
            line_edit.setReadOnly(True)
            line_edit.setStyleSheet(READ_ONLY_STYLE)
        return line_edit

    @staticmethod
    def _as_string(value) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    # --- редактирование ---

    def toggle_edit_block(self, fields, button: QPushButton) -> None:
        """Переключает блок между режимом просмотра и правки."""
        is_editing = button.text() == "Редактировать"
        for widget in self._block_widgets(fields):
            if isinstance(widget, QLineEdit):
                widget.setReadOnly(not is_editing)
                widget.setStyleSheet(EDITABLE_STYLE if is_editing else READ_ONLY_STYLE)
            elif isinstance(widget, QComboBox):
                widget.setEnabled(is_editing)
                widget.setStyleSheet(EDITABLE_STYLE if is_editing else READ_ONLY_STYLE)

        button.setText("Сохранить" if is_editing else "Редактировать")
        if not is_editing:
            self._save_block_data(fields)

    def _block_widgets(self, fields) -> list[QWidget]:
        widgets = []
        for _, field_index in fields:
            widget = self.field_widgets.get(
                self.widget_key(COLUMN_INDEX_TO_FIELD_NAME.get(field_index))
            )
            if widget is not None:
                widgets.append(widget)
        return widgets

    def _save_block_data(self, fields) -> None:
        """Пишет заполненные поля блока в таблицу ``eyes``."""
        eye_id = self._current_eye_id()
        if eye_id is None:
            logger.warning("Не найден id записи eyes — обновление пропущено")
            return

        values = self._block_values(fields)
        if not values:
            return

        set_clause = ", ".join(f"{field} = :{field}" for field in values)
        query_text = f"UPDATE {EYE_TABLE_NAME} SET {set_clause} WHERE id = :eye_id"

        query = QSqlQuery(self.db_connection)
        query.prepare(query_text)
        for field, value in values.items():
            query.bindValue(f":{field}", value)
        query.bindValue(":eye_id", eye_id)
        log_sql(query_text, query, values.keys(), {"eye_id": eye_id})

        if not query.exec():
            logger.error("Не удалось обновить eyes: %s", query.lastError().text())

    def _block_values(self, fields) -> dict[str, object]:
        values: dict[str, object] = {}
        for _, field_index in fields:
            field_name = COLUMN_INDEX_TO_FIELD_NAME.get(field_index)
            widget_key = self.widget_key(field_name)
            if self.field_widgets.get(widget_key) is None:
                continue
            value = self.get_field_value(widget_key)
            if value is not None and str(value).strip():
                values[field_name] = value
        return values

    def _current_eye_id(self):
        if self.row_index is None or self.appointment_data_model is None:
            return None
        return self.appointment_data_model.index(self.row_index, 0).data()

    # --- доступ к полям ---

    def get_field_value(self, widget_key: str):
        widget = self.field_widgets.get(widget_key)
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def get_all_field_widgets(self) -> dict[str, QWidget]:
        return self.field_widgets

    def clear_all_fields(self) -> None:
        for widget in self.field_widgets.values():
            if isinstance(widget, QLineEdit):
                widget.clear()
                widget.setReadOnly(False)
                widget.setStyleSheet(EDITABLE_STYLE)
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(-1)
                widget.setEnabled(True)
