from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlQuery
from PySide6.QtWidgets import QLineEdit, QComboBox, QPushButton, QGroupBox, QFormLayout, QWidget, QVBoxLayout

from bsmu.macula.plugins.db.constants import DROPDOWN_DB_VALUES, DROPDOWN_DISPLAY_MAP, BLOCKS, \
    COLUMN_INDEX_TO_FIELD_NAME


class DynamicFormBuilder(QWidget):
    def __init__(self, blocks, initial_data, eye_index, appointment_data_model, db_manager, patient_id_clicked, parent=None):
        super().__init__(parent)
        self.eye_index  = eye_index
        self.appointment_data_model  = appointment_data_model
        self.db_manager  = db_manager
        self.patient_id_clicked = patient_id_clicked
        self.dropdown_db_values = DROPDOWN_DB_VALUES

        self.dropdown_display_map = DROPDOWN_DISPLAY_MAP
        self.main_layout = QVBoxLayout(self)
        self.field_widgets = {}
        self.initial_data = initial_data or {}

        self.build_blocks(blocks, eye_index)

    # def insert_new_eye_record(self, eye_index, patient_id_clicked, appointment_data_model, db_manager):
    #     table_name = "eyes"
    #     all_fields = []
    #     all_values = {}
    #     blocks = BLOCKS
    #     # Сбор данных из всех виджетов
    #     for block_title, fields in blocks:
    #         for label, field_name in fields:
    #             widget = self.formDict[eye_index].get_all_field_widgets().get(field_name + str(eye_index))
    #             if widget is None:
    #                 continue
    #
    #             # Получение значения
    #             if isinstance(widget, QComboBox):
    #                 value = widget.currentData()
    #             elif isinstance(widget, QLineEdit):
    #                 value = widget.text()
    #             else:
    #                 continue
    #
    #             all_fields.append(field_name)
    #             if value is not None and str(value).strip():
    #                 all_values[field_name] = value
    #
    #             print(f"{field_name} = '{value}'")  # отладка
    #
    #     # Проверка на наличие данных
    #     if not all_fields:
    #         print("❌ Нет данных для вставки.")
    #         return
    #
    #     # Вставка в appointments
    #     query1 = QSqlQuery(db_manager.get_connection())
    #     query1.prepare("""
    #         INSERT INTO appointments (pacient_id, date, duration_of_the_disease)
    #         VALUES (:pacient_id, :date2, :duration_of_the_disease2)
    #     """)
    #     query1.bindValue(":pacient_id", patient_id_clicked)
    #     query1.bindValue(":date2", all_values.get("date"))
    #     query1.bindValue(":duration_of_the_disease2", all_values.get("duration_of_the_disease"))
    #
    #     if not query1.exec_():
    #         print("❌ Ошибка при вставке appointments:", query1.lastError().text())
    #         return
    #
    #     appointment_id = query1.lastInsertId()
    #     print(f"✅ Создана запись appointments с ID: {appointment_id}")
    #
    #     # Добавляем служебные поля
    #     all_fields += ["appointment_id", "eye", "MKO", "topkon", "optopol"]
    #     all_values["appointment_id"] = appointment_id
    #     all_values["eye"] = eye_index
    #     all_values["MKO"] = eye_index
    #     all_values["topkon"] = True
    #     all_values["optopol"] = True
    #
    #     # Формируем SQL-запрос
    #     field_clause = ", ".join(all_fields)
    #     placeholder_clause = ", ".join([f":{field}" for field in all_fields])
    #     query_text = f"INSERT INTO {table_name} ({field_clause}) VALUES ({placeholder_clause})"
    #
    #     query2 = QSqlQuery(db_manager.get_connection())
    #     query2.prepare(query_text)
    #
    #     for field, value in all_values.items():
    #         query2.bindValue(f":{field}", value)
    #
    #     if not query2.exec_():
    #         print("❌ Ошибка при вставке eyes:", query2.lastError().text())
    #     else:
    #         print("✅ Запись в eyes успешно добавлена.")
    #         # self._load_appointments(patient_id_clicked)

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
                field_name = self._get_column_name(field_name2) + str(eye_index)
                value = self.initial_data.get(field_name)
                widget = self.create_field_widget(field_name, value, has_initial_values)
                form_layout.addRow(label, widget)
                self.field_widgets[field_name] = widget

            self.main_layout.addWidget(group_box)

            if has_initial_values:
                edit_button = QPushButton("Редактировать")
                edit_button.clicked.connect(lambda _, g=fields, b=edit_button: self.toggle_edit_block(g, b))
                form_layout.addRow("", edit_button)

    # def _toggle_edit_mode_2(self, button: QPushButton, eye_index: int):
    #     is_editing = button.text() == "Редактировать"
    #
    #     # Переключаем все поля
    #     for field_name, widget in self.field_widgets.items():
    #         if isinstance(widget, QLineEdit):
    #             widget.setReadOnly(not is_editing)
    #             widget.setStyleSheet(
    #                 "background-color: #ffffff;" if is_editing else "background-color: #f0f0f0;"
    #             )
    #         elif isinstance(widget, QComboBox):
    #             widget.setEnabled(is_editing)
    #
    #     # Переключаем текст кнопки
    #     if is_editing:
    #         button.setText("Сохранить")
    #     else:
    #         button.setText("Редактировать")
    #         # Вставка новой записи
    #         self.insert_new_eye_record(
    #             eye_index,
    #             self.patient_id_clicked,
    #             self.appointment_data_model,
    #             self.db_manager
    #         )

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
        eye_id_value = self.appointment_data_model.index(self.eye_index, 0).data()

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
            combo = QComboBox()
            db_values = self.dropdown_db_values[field_name]
            display_map = self.dropdown_display_map[field_name]

            for db_value in db_values:
                display_text = display_map.get(db_value, db_value)
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

    def get_all_values(self):
        return {
            field: self.get_field_value(field)
            for field in self.field_widgets
        }
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