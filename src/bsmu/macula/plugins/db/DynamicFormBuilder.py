from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem, QFont
from PySide6.QtSql import QSqlQuery
import pandas as pd
from PySide6.QtWidgets import QLineEdit, QComboBox, QPushButton, QGroupBox, QFormLayout, QWidget, QVBoxLayout, \
    QTableView, QLabel

from bsmu.macula.plugins.db.HoverComboBox import HoverComboBox
from bsmu.macula.plugins.db.WheelBlocker import WheelBlocker
from bsmu.macula.plugins.db.constants import DROPDOWN_DB_VALUES, DROPDOWN_DISPLAY_MAP, BLOCKS, \
    COLUMN_INDEX_TO_FIELD_NAME, DROPDOWN_DISPLAY_MAP2


class DynamicFormBuilder(QWidget):
    def __init__(self, blocks, initial_data, eye_index, appointment_data_model, db_manager, patient_id_clicked,rowInt = None,
                 parent=None):
        super().__init__(parent)
        self.eye_index = eye_index
        self.rowInt = rowInt
        self.appointment_data_model = appointment_data_model
        self.db_manager = db_manager
        self.patient_id_clicked = patient_id_clicked
        self.dropdown_db_values = DROPDOWN_DB_VALUES

        self.dropdown_display_map = DROPDOWN_DISPLAY_MAP
        self.dropdown_display_map2 = DROPDOWN_DISPLAY_MAP2
        self.main_layout = QVBoxLayout(self)
        self.field_widgets = {}
        self.initial_data = initial_data or {}

        self.build_blocks(blocks, eye_index)

    def get_column_index_by_name(self, model, column_name: str) -> int:
        """Возвращает индекс столбца по его названию"""
        for col in range(model.columnCount()):
            header = model.headerData(col, Qt.Horizontal)
            if header == column_name:
                return col
        return -1  # если не найдено

    def build_blocks(self, blocks, eye_index):
        has_initial_values = self.initial_data.get('date0') or self.initial_data.get('date1')
        for block_title, fields in blocks:
            group_box = QGroupBox(block_title) if block_title else QWidget()
            form_layout = QFormLayout(group_box)

            for label, field_name2 in fields:
                field_name = self._get_column_name(field_name2) + str(eye_index if eye_index is not None else "_")
                value = self.initial_data.get(field_name)
                widget = self.create_field_widget(field_name, value, has_initial_values)
                form_layout.addRow(label, widget)
                self.field_widgets[field_name] = widget

            self.main_layout.addWidget(group_box)

            if has_initial_values:
                edit_button = QPushButton("Редактировать")
                edit_button.clicked.connect(lambda _, g=fields, b=edit_button: self.toggle_edit_block(g, b))
                form_layout.addRow("", edit_button)

        # if (self.eye_index is not None):
        #     fields_injections = ["avastin_injections",
        #     "eylea_injections",
        #     "visque_injections",
        #     "diprospan_injections",
        #     "kenalog_injections",
        #     "lucentis_injections"]
        #     values_injections = self.extract_fields_from_model(self.appointment_data_model, fields_injections, self.eye_index)
        #     print("Значения:", values_injections)
        #     fields_names = ["avastin",
        #     "eylea",
        #     "visque",
        #     "diprospan",
        #     "kenalog",
        #     "lucentis"]
        #     values_names = self.extract_fields_from_model(self.appointment_data_model, fields_names, self.eye_index)
        #     print("Значения:", values_names)
        #     injections_data = [
        #         {"eye_id": 0, "lutein_therapy": "бевацизумаб (Авастин)"},
        #         {"eye_id": 1, "lutein_therapy": "афлиберцепт (Эйлеа)"},
        #         {"eye_id": 2, "lutein_therapy": "бролоцизумаб (Визкью)"},
        #         {"eye_id": 3, "lutein_therapy": "Бетаметазон (Дипроспан)"},
        #         {"eye_id": 4, "lutein_therapy": "триамциналон (Кеналог)"},
        #         {"eye_id": 5, "lutein_therapy": "Луцентис"}
        #     ]
        #     title = QLabel("💉 Только ненулевые значения")
        #     title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        #     title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #     self.main_layout.addWidget(title)
        #
        #     table = QTableView()
        #     table.setMinimumHeight(250)
        #     self.show_filtered_injections(table, values_names, values_injections, injections_data)
        #     self.main_layout.addWidget(table)

    def show_filtered_injections(self, table_view, values_names, values_injections, injections_data):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Препарат", "Количество"])

        for i in range(len(values_injections)):
            count = int(values_injections[i]) if values_injections[i] else 0
            name = int(values_names[i]) if values_names[i] else 0
            label = injections_data[i]["lutein_therapy"]

            if count != 0 or name != 0:  # Показываем, если есть хоть что-то
                items = [
                    QStandardItem(label),
                    QStandardItem(str(count))
                ]
                for item in items:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                model.appendRow(items)

        table_view.setModel(model)
        table_view.setAlternatingRowColors(True)
        table_view.setStyleSheet("""
            QTableView {
                font-size: 12pt;
                gridline-color: #ccc;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                font-weight: bold;
            }
        """)
        table_view.resizeColumnsToContents()

    def extract_fields_from_model(self, model, field_names, row_index):
        values = []
        for field in field_names:
            column_index = None
            for col in range(model.columnCount()):
                header = model.headerData(col, Qt.Orientation.Horizontal)
                if header == field:
                    column_index = col
                    break
            if column_index is None:
                raise ValueError(f"Поле '{field}' не найдено в модели.")
            index = model.index(row_index, column_index)
            values.append(model.data(index))
        return values

    def toggle_edit_block(self, fields, button):
        is_editing = button.text() == "Редактировать"

        for _, field_name2 in fields:
            field_name = self._get_column_name(field_name2)
            widget = self.field_widgets.get(field_name + str(self.eye_index))
            if isinstance(widget, QLineEdit):
                widget.setReadOnly(not is_editing)
                widget.setStyleSheet("background-color: #ffffff;" if is_editing else "background-color: #f0f0f0;")
            elif isinstance(widget, QComboBox):
                widget.setEnabled(is_editing)
                widget.setStyleSheet("background-color: #ffffff;" if is_editing else "background-color: #f0f0f0;")

        if is_editing:
            button.setText("Сохранить")
        else:
            button.setText("Редактировать")
            # вызов сохранения
            self._save_block_data(fields)

    def toggle_add_appointment_dialog(self, fields, button):
        is_editing = button.text() == "Редактировать"

        for _, field_name2 in fields:
            field_name = self._get_column_name(field_name2)
            widget = self.field_widgets.get(field_name + str(self.eye_index))
            if isinstance(widget, QLineEdit):
                widget.setReadOnly(not is_editing)
                widget.setStyleSheet("background-color: #ffffff;" if is_editing else "background-color: #f0f0f0;")
            elif isinstance(widget, QComboBox):
                widget.setEnabled(is_editing)
                widget.setStyleSheet("background-color: #ffffff;" if is_editing else "background-color: #f0f0f0;")

        if is_editing:
            button.setText("Сохранить")
        else:
            button.setText("Редактировать")
            # вызов сохранения
            self._save_block_data(fields)

    def _save_block_data(self, fields):
        table_name = "eyes"
        values = {}
        fields_to_update = []

        for _, field_name2 in fields:
            field_name = self._get_column_name(field_name2)
            widget = self.field_widgets.get(field_name + str(self.eye_index))
            if widget is None:
                continue

            # Получение значения из виджета
            if isinstance(widget, QComboBox):
                value = widget.currentData()
            elif isinstance(widget, QLineEdit):
                value = widget.text()
            else:
                continue

            if value is not None and str(value).strip():
                fields_to_update.append(field_name)
                values[field_name] = value

                # Обновление модели
                column_index = self.get_column_index_by_name(self.appointment_data_model, field_name)
                index = self.appointment_data_model.index(self.eye_index, column_index)
                if index.isValid():
                    self.appointment_data_model.setData(index, value)

        # Получение ID записи
        eye_id_name = "id"
        eye_id_value = self.appointment_data_model.index(self.rowInt, 0).data()

        if fields_to_update and values and eye_id_value is not None:
            set_clause = ", ".join([f"{field} = :{field}" for field in fields_to_update])
            query_text = f"""
                UPDATE {table_name}
                SET {set_clause}
                WHERE {eye_id_name} = :{eye_id_name}
            """

            query = QSqlQuery(self.db_manager)
            query.prepare(query_text)

            # Привязка значений
            for field in fields_to_update:
                value = values[field]
                column_index = self.get_column_index_by_name(self.appointment_data_model, field)
                index = self.appointment_data_model.index(self.eye_index, column_index)
                raw_data = index.data() if index.isValid() else None
                data_type = type(raw_data) if raw_data is not None else str

                try:
                    converted_value = data_type(value) if callable(data_type) else value
                except Exception as e:
                    print(f"❌ Ошибка преобразования поля {field}: {e}")
                    converted_value = value

                query.bindValue(f":{field}", converted_value)

            query.bindValue(f":{eye_id_name}", eye_id_value)

            # Отладка
            print("📤 SQL-запрос:")
            print(query_text)
            print("📦 Параметры:")
            for field in fields_to_update:
                print(f"{field} → {query.boundValue(f':{field}')}")
            print(f"{eye_id_name} → {eye_id_value}")

            # Выполнение
            if not query.exec_():
                print(f"❌ Ошибка при обновлении: {query.lastError().text()}")
            else:
                print(f"✅ Данные блока успешно обновлены.")

    def _get_column_name(self, column_index: int) -> str:
        """Возвращает имя поля по индексу"""
        column_index_to_field_name = COLUMN_INDEX_TO_FIELD_NAME

        # maybe_name = self.appointment_data_model.headerData(column_index, Qt.Horizontal) надо разобраться
        return column_index_to_field_name.get(column_index)

    def create_field_widget(self, field_name2, initial_value=None, disable=True):
        field_name = field_name2[:-1]
        if field_name in self.dropdown_db_values:
            combo = HoverComboBox()
            db_values = self.dropdown_db_values[field_name]
            display_map = self.dropdown_display_map[field_name]
            display_map2 = self.dropdown_display_map2.get(field_name)
            combo.installEventFilter(WheelBlocker(combo))
            combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            for db_value in db_values:
                display_text = display_map.get(db_value, db_value)
                if (display_map2 is None):
                    combo.addItem(display_text, db_value)
                else:
                    display_text2 = display_map2.get(db_value, db_value)
                    if (display_text2.strip()):
                        combo.addItem(display_text, db_value, display_text2)
                    else:
                        combo.addItem(display_text, db_value)


            def to_str(value):
                if value is None:
                    return ""
                if isinstance(value, float):
                    return f"{value:.2f}"
                return str(value)

            t = to_str(initial_value)
            if t in db_values:
                index = db_values.index(t)
                combo.setCurrentIndex(index)
                if (disable):
                    combo.setEnabled(False)
                    combo.setStyleSheet("background-color: #f0f0f0;")

            return combo
        else:
            line_edit = QLineEdit()
            if initial_value is not None:
                line_edit.setText(str(initial_value))
            if (disable):
                line_edit.setReadOnly(True)
                line_edit.setStyleSheet("background-color: #f0f0f0;")
            return line_edit

    def get_field_value(self, field_name):
        widget = self.field_widgets.get(field_name)
        if isinstance(widget, QComboBox):
            return widget.currentData()
        elif isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def get_all_field_widgets(self):
        return self.field_widgets

    def clear_all_fields(self):
        for field_name, widget in self.field_widgets.items():
            if isinstance(widget, QLineEdit):
                widget.clear()
                widget.setStyleSheet("background-color: #ffffff;")
                widget.setReadOnly(False)
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(-1)  # сброс выбора
                widget.setEnabled(True)
