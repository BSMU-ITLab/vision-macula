from functools import partial

from PySide6.QtCore import Qt, QModelIndex
from PySide6.QtSql import QSqlQueryModel, QSqlQuery
from bsmu.macula.plugins.db.edit_pacirnt_delegate import EditPacientDelegate
from PySide6.QtWidgets import (
    QWidget, QTableView, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QGroupBox, QFormLayout, QLineEdit,
    QTabWidget, QMessageBox, QComboBox
)
import bsmu.macula.plugins.db.images.dbicons_rc

from bsmu.macula.plugins.db.add_patient_dialog import AddRecordDialog
from bsmu.macula.plugins.db.database_manager_v2 import DatabaseManager


class DynamicFormBuilder(QWidget):
    def __init__(self, blocks, initial_data, eye_index, appointment_data_model, db_manager, patient_id_clicked, parent=None):
        super().__init__(parent)
        self.eye_index  = eye_index
        self.appointment_data_model  = appointment_data_model
        self.db_manager  = db_manager
        self.patient_id_clicked = patient_id_clicked
        self.dropdown_db_values = {
            "areds": ["", "1", "2", "3", "4", "4a", "4b", "4c", "4d"],
            "refraction": ["", "Мсл", "Мср", "Em", "Hmсл", "Hmср", "МКОЗ"],
            "rpe_status": ["", "1", "2", "3", "4", "5"],
            "rpe_localisation": ["", "1", "2", "3", "4", "5"],
            "cme_localisation": ["", "0", "1", "2", "3", "4"],
            "serouz_rpe_detachment_localisation": ["", "0", "1", "2", "3", "4"],
            "hemorrhagic_rpe_detachment_localisation": ["", "0", "1", "2", "3", "4"],
            "fibrovascular_rpe_detachment_localisation": ["", "0", "1", "2", "3", "4"],
            "drusenoid_detachment_rpe_localisation": ["", "0", "1", "2", "3", "4"],
            "druses_localisation": ["", "0", "1", "2", "3", "4"],
            "fluid_under_rpe_localisation": ["", "1", "2", "3", "4"],
            "ez_status": ["", "1", "2", "3", "4"],
            "ez_localisation": ["", "1", "2", "3", "4"],
            "myoidnz_status": ["", "1", "2", "3"],
            "myoidnz_localisation": ["", "1", "2", "3", "4"],
            "rne_detachment_localisation": ["", "0", "1", "2", "3", "4"],
            "hyperreflective_material_localisation": ["", "0", "1", "2"]
        }

        self.dropdown_display_map = {
            "areds": {
                "": "",
                "1": "1 ст (AREDS 1) — отсутствие ВМД или мелкие друзы <63μm",
                "2": "2 ст (AREDS 2) — множественные мелкие и немного средних друз",
                "3": "3 ст (AREDS 3) — множество средних, ≥1 крупная друза или ГА вне фовеолы",
                "4": "4 ст (AREDS 4) — ГА в фовеоле или неоваскулярная макулопатия",
                "4a": "4а — географическая атрофия",
                "4b": "4в — хориоидальная неоваскуляризация (ХНВ 1,2, РАП, ПХВ)",
                "4c": "4с — субретинальная и подПЭС фиброваскулярная пролиферация",
                "4d": "4d — дисковидный рубец"
            },
            "refraction": {
                "": "",
                "Мсл": "Мсл — миопия слабой степени",
                "Мср": "Мср — миопия средней степени",
                "Em": "Em — эмметропия",
                "Hmсл": "Hmсл — гиперметропия слабой степени",
                "Hmср": "Hmср — гиперметропия средней степени",
                "МКОЗ": "МКОЗ — максимальная корригированная острота зрения"
            },
            "rpe_status": {
                "": "",
                "1": "1 — эпителий сохранён",
                "2": "2 — эпителий неравномерный",
                "3": "3 — единичные разрывы",
                "4": "4 — множественные разрывы",
                "5": "5 — эпителий не определяется"
            },
            "rpe_localisation": {
                "": "",
                "1": "1 — фовеола",
                "2": "2 — фовеа (без фовеолы)",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула (без фовеа и фовеолы)",
                "5": "5 — autofmacula (без изменений)"
            },
            "cme_localisation": {
                "": "",
                "0": "0 — отсутствие отёка",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "serouz_rpe_detachment_localisation": {
                "": "",
                "0": "0 — отсутствие отслойки",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "hemorrhagic_rpe_detachment_localisation": {
                "": "",
                "0": "0 — отсутствие отслойки",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "fibrovascular_rpe_detachment_localisation": {
                "": "",
                "0": "0 — отсутствие отслойки",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "drusenoid_detachment_rpe_localisation": {
                "": "",
                "0": "0 — отсутствие отслойки",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "druses_localisation": {
                "": "",
                "0": "0 — отсутствие друз",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "fluid_under_rpe_localisation": {
                "": "",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "ez_status": {
                "": "",
                "1": "1 — сохранена",
                "2": "2 — неравномерная (фрагментация)",
                "3": "3 — не определяется локально",
                "4": "4 — не определяется"
            },
            "ez_localisation": {
                "": "",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "myoidnz_status": {
                "": "",
                "1": "1 — сохранена",
                "2": "2 — неравномерная (фрагментация)",
                "3": "3 — не определяется"
            },
            "myoidnz_localisation": {
                "": "",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "rne_detachment_localisation": {
                "": "",
                "0": "0 — отсутствие отслойки",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "hyperreflective_material_localisation": {
                "": "",
                "0": "0 — отсутствие",
                "1": "1 — субретинальный",
                "2": "2 — интаретинальный"
            }
        }
        self.main_layout = QVBoxLayout(self)
        self.field_widgets = {}
        self.initial_data = initial_data or {}

        self.build_blocks(blocks, eye_index)

    def insert_new_eye_record(self, eye_index, patient_id_clicked, appointment_data_model, db_manager):
        table_name = "eyes"
        all_fields = []
        all_values = {}
        blocks = [
            ("Обследование", [
                ("Дата посещения", 3),
                ("Продолжительность заболевания", 4),
                ("Тип томографа", 5),
                ("Критерий AREDS", 6),
                ("Рефракция", 7),
                ("Тип неоваскуляризации", 8)
            ]),
            ("Ретинальные показатели", [
                ("Толщина хориоидеи в центре", 9),
                ("Толщина сетчатки в фовеоле", 10),
                ("Общий объем", 11),
                ("Средний объем", 12)
            ]),
            ("", [
                ("Состояние РПЭ", 13),
                ("Локализация дефектов РПЭ", 14),
                ("Локализация кистозного макулярного отека", 15)
            ]),
            ("Серозная ОПЭ", [
                ("Локализация", 16),
                ("Ширина", 17),
                ("Высота", 18),
                ("Площадь", 19)
            ]),
            ("Геморрагическая ОПЭ", [
                ("Локализация", 20),
                ("Ширина", 21),
                ("Высота", 22),
                ("Площадь", 23)
            ]),
            ("Фиброваскулярная ОПЭ", [
                ("Локализация", 24),
                ("Ширина", 25),
                ("Высота", 26),
                ("Площадь", 27)
            ]),
            ("Друзеноидная ОПЭ", [
                ("Локализация", 28),
                ("Ширина", 29),
                ("Высота", 30),
                ("Площадь", 31)
            ]),
            ("Друзы", [
                ("Локализация", 32),
                ("Ширина", 33),
                ("Высота", 34),
                ("Площадь", 35)
            ]),
            ("Жидкость под РПЭ", [
                ("Пощадь", 36),
                ("Локализация", 37)
            ]),
            ("Эллипсоидная зона", [
                ("Состояние", 38),
                ("Локализация дефектов", 39)
            ]),
            ("Миоидная зона", [
                ("Состояние", 40),
                ("Локализация дефектов", 41)
            ]),
            ("Отслойка нейросенсорной сетчатки", [
                ("Локализация", 42),
                ("Ширина", 43),
                ("Высота", 44),
                ("Площадь", 45)
            ]),
            ("Гиперрефлективный материал", [
                ("Локализация", 46),
                ("Площадь", 47)
            ])]

        # Сбор данных из всех виджетов
        for block_title, fields in blocks:
            for label, field_name in fields:
                widget = self.formDict[eye_index].get_all_field_widgets().get(field_name + str(eye_index))
                if widget is None:
                    continue

                # Получение значения
                if isinstance(widget, QComboBox):
                    value = widget.currentData()
                elif isinstance(widget, QLineEdit):
                    value = widget.text()
                else:
                    continue

                all_fields.append(field_name)
                if value is not None and str(value).strip():
                    all_values[field_name] = value

                print(f"{field_name} = '{value}'")  # отладка

        # Проверка на наличие данных
        if not all_fields:
            print("❌ Нет данных для вставки.")
            return

        # Вставка в appointments
        query1 = QSqlQuery(db_manager.get_connection())
        query1.prepare("""
            INSERT INTO appointments (pacient_id, date, duration_of_the_disease)
            VALUES (:pacient_id, :date2, :duration_of_the_disease2)
        """)
        query1.bindValue(":pacient_id", patient_id_clicked)
        query1.bindValue(":date2", all_values.get("date"))
        query1.bindValue(":duration_of_the_disease2", all_values.get("duration_of_the_disease"))

        if not query1.exec_():
            print("❌ Ошибка при вставке appointments:", query1.lastError().text())
            return

        appointment_id = query1.lastInsertId()
        print(f"✅ Создана запись appointments с ID: {appointment_id}")

        # Добавляем служебные поля
        all_fields += ["appointment_id", "eye", "MKO", "topkon", "optopol"]
        all_values["appointment_id"] = appointment_id
        all_values["eye"] = eye_index
        all_values["MKO"] = eye_index
        all_values["topkon"] = True
        all_values["optopol"] = True

        # Формируем SQL-запрос
        field_clause = ", ".join(all_fields)
        placeholder_clause = ", ".join([f":{field}" for field in all_fields])
        query_text = f"INSERT INTO {table_name} ({field_clause}) VALUES ({placeholder_clause})"

        query2 = QSqlQuery(db_manager.get_connection())
        query2.prepare(query_text)

        for field, value in all_values.items():
            query2.bindValue(f":{field}", value)

        if not query2.exec_():
            print("❌ Ошибка при вставке eyes:", query2.lastError().text())
        else:
            print("✅ Запись в eyes успешно добавлена.")
            # self._load_appointments(patient_id_clicked)

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

    def _toggle_edit_mode_2(self, button: QPushButton, eye_index: int):
        is_editing = button.text() == "Редактировать"

        # Переключаем все поля
        for field_name, widget in self.field_widgets.items():
            if isinstance(widget, QLineEdit):
                widget.setReadOnly(not is_editing)
                widget.setStyleSheet(
                    "background-color: #ffffff;" if is_editing else "background-color: #f0f0f0;"
                )
            elif isinstance(widget, QComboBox):
                widget.setEnabled(is_editing)

        # Переключаем текст кнопки
        if is_editing:
            button.setText("Сохранить")
        else:
            button.setText("Редактировать")
            # Вставка новой записи
            self.insert_new_eye_record(
                eye_index,
                self.patient_id_clicked,
                self.appointment_data_model,
                self.db_manager
            )

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

    def toggle_edit_all(self):
        for field_name, widget in self.field_widgets.items():
            if isinstance(widget, QLineEdit):
                widget.setReadOnly(False)
                widget.setStyleSheet("background-color: #ffffff;")
            elif isinstance(widget, QComboBox):
                widget.setEnabled(True)

        self.defer_edit_button.setText("Сохранить всё")
        self.defer_edit_button.clicked.disconnect()
        self.defer_edit_button.clicked.connect(lambda: self.save_all_and_lock())

    def save_all_and_lock(self):
        self.defer_edit_button.setText("Редактировать всё")
        for field_name, widget in self.field_widgets.items():
            if isinstance(widget, QLineEdit):
                widget.setReadOnly(True)
                widget.setStyleSheet("background-color: #f0f0f0;")
            elif isinstance(widget, QComboBox):
                widget.setEnabled(False)

        self.defer_edit_button.clicked.disconnect()
        self.defer_edit_button.clicked.connect(lambda: self.toggle_edit_all())
        self._save_block_data(list(self.field_widgets.items()))

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
        column_index_to_field_name = {
            3: "date",
            4: "duration_of_the_disease",
            5: "topkon",
            6: "areds",
            7: "refraction",
            8: "type_of_neovascularization",
            9: "choroidal_thickness_center",
            10: "cts_foveola",
            11: "total_volume",
            12: "average_volume",
            13: "rpe_status",
            14: "rpe_localisation",
            15: "cme_localisation",
            16: "serouz_rpe_detachment_localisation",
            17: "serouz_rpe_detachment_width",
            18: "serouz_rpe_detachment_height",
            19: "serouz_rpe_detachment_area",
            20: "hemorrhagic_rpe_detachment_localisation",
            21: "hemorrhagic_rpe_detachment_width",
            22: "hemorrhagic_rpe_detachment_heidgt",
            23: "hemorrhagic_rpe_detachment_area",
            24: "fibrovascular_rpe_detachment_localisation",
            25: "fibrovascular_rpe_detachment_width",
            26: "fibrovascular_rpe_detachment_heidgt",
            27: "fibrovascular_rpe_detachment_area",
            28: "drusenoid_detachment_rpe_localisation",
            29: "drusenoid_detachment_rpe_width",
            30: "drusenoid_detachment_rpe_height",
            31: "drusenoid_detachment_rpe_area",
            32: "druses_localisation",
            33: "druses_weigt",
            34: "druses_heigt",
            35: "dzuses_area",
            36: "fluid_under_rpe_area",
            37: "fluid_under_rpe_localisation",
            38: "ez_status",
            39: "ez_localisation",
            40: "myoidnz_status",
            41: "myoidnz_localisation",
            42: "rne_detachment_localisation",
            43: "rne_detachment_width",
            44: "rne_detachment_heigt",
            45: "rne_detachment_area",
            46: "hyperreflective_material_localisation",
            47: "hyperreflective_material_area"
        }

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

class PatientsModel(QSqlQueryModel):
    def columnCount(self, parent=QModelIndex()):
        return super().columnCount(parent) + 1  # +1 для кнопки

    def data(self, index, role=Qt.DisplayRole):
        if index.column() == super().columnCount():
            if role == Qt.DisplayRole:
                return ""  # Пусто, но ячейка существует
        return super().data(index, role)

class TableWidgetExample(QWidget):



    def __init__(self, bd_path):
        super().__init__()
        self.formDict = {}
        self.dropdown_db_values = {
            "areds": ["1", "2", "3", "4", "4a", "4b", "4c", "4d"],
            "refraction": ["Мсл", "Мср", "Em", "Hmсл", "Hmср", "МКОЗ"],
            "rpe_status": ["1", "2", "3", "4", "5"],
            "rpe_localisation": ["1", "2", "3", "4", "5"],
            "cme_localisation": ["0", "1", "2", "3", "4"],
            "serouz_rpe_detachment_localisation": ["0", "1", "2", "3", "4"],
            "hemorrhagic_rpe_detachment_localisation": ["0", "1", "2", "3", "4"],
            "fibrovascular_rpe_detachment_localisation": ["0", "1", "2", "3", "4"],
            "drusenoid_detachment_rpe_localisation": ["0", "1", "2", "3", "4"],
            "druses_localisation": ["0", "1", "2", "3", "4"],
            "fluid_under_rpe_localisation": ["1", "2", "3", "4"],
            "ez_status": ["1", "2", "3", "4"],
            "ez_localisation": ["1", "2", "3", "4"],
            "myoidnz_status": ["1", "2", "3"],
            "myoidnz_localisation": ["1", "2", "3", "4"],
            "rne_detachment_localisation": ["0", "1", "2", "3", "4"],
            "hyperreflective_material_localisation": ["0", "1", "2"]
        }

        self.dropdown_display_map = {
            "areds": {
                "1": "1 ст (AREDS 1) — отсутствие ВМД или мелкие друзы <63μm",
                "2": "2 ст (AREDS 2) — множественные мелкие и немного средних друз",
                "3": "3 ст (AREDS 3) — множество средних, ≥1 крупная друза или ГА вне фовеолы",
                "4": "4 ст (AREDS 4) — ГА в фовеоле или неоваскулярная макулопатия",
                "4a": "4а — географическая атрофия",
                "4b": "4в — хориоидальная неоваскуляризация (ХНВ 1,2, РАП, ПХВ)",
                "4c": "4с — субретинальная и подПЭС фиброваскулярная пролиферация",
                "4d": "4d — дисковидный рубец"
            },
            "refraction": {
                "Мсл": "Мсл — миопия слабой степени",
                "Мср": "Мср — миопия средней степени",
                "Em": "Em — эмметропия",
                "Hmсл": "Hmсл — гиперметропия слабой степени",
                "Hmср": "Hmср — гиперметропия средней степени",
                "МКОЗ": "МКОЗ — максимальная корригированная острота зрения"
            },
            "rpe_status": {
                "1": "1 — эпителий сохранён",
                "2": "2 — эпителий неравномерный",
                "3": "3 — единичные разрывы",
                "4": "4 — множественные разрывы",
                "5": "5 — эпителий не определяется"
            },
            "rpe_localisation": {
                "1": "1 — фовеола",
                "2": "2 — фовеа (без фовеолы)",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула (без фовеа и фовеолы)",
                "5": "5 — autofmacula (без изменений)"
            },
            "cme_localisation": {
                "0": "0 — отсутствие отёка",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "serouz_rpe_detachment_localisation": {
                "0": "0 — отсутствие отслойки",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "hemorrhagic_rpe_detachment_localisation": {
                "0": "0 — отсутствие отслойки",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "fibrovascular_rpe_detachment_localisation": {
                "0": "0 — отсутствие отслойки",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "drusenoid_detachment_rpe_localisation": {
                "0": "0 — отсутствие отслойки",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "druses_localisation": {
                "0": "0 — отсутствие друз",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "fluid_under_rpe_localisation": {
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "ez_status": {
                "1": "1 — сохранена",
                "2": "2 — неравномерная (фрагментация)",
                "3": "3 — не определяется локально",
                "4": "4 — не определяется"
            },
            "ez_localisation": {
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "myoidnz_status": {
                "1": "1 — сохранена",
                "2": "2 — неравномерная (фрагментация)",
                "3": "3 — не определяется"
            },
            "myoidnz_localisation": {
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "rne_detachment_localisation": {
                "0": "0 — отсутствие отслойки",
                "1": "1 — фовеола",
                "2": "2 — фовеа",
                "3": "3 — фовеола + фовеа",
                "4": "4 — макула"
            },
            "hyperreflective_material_localisation": {
                "0": "0 — отсутствие",
                "1": "1 — субретинальный",
                "2": "2 — интаретинальный"
            }
        }
        try:
            self.db_path = bd_path
            self.db_manager = DatabaseManager(bd_path)
            self.db = self.db_manager.get_connection()
            self.layoutBoxes = QVBoxLayout()
            self._setup_ui()
            self._load_initial_data()
        except ConnectionError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            self.close()

    def _setup_ui(self):
        self.appointment_data_model = QSqlQueryModel()
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("Пациенты")
        self.resize(1200, 800)

        self.main_layout = QHBoxLayout(self)
        self.left_layout = QVBoxLayout()

        # Таблица пациентов
        self.patients_model = PatientsModel()
        self._setup_patients_table()

        # Таблица приемов
        self.appointments_model = QSqlQueryModel()
        self._setup_appointments_table()

        # Вкладки с детальной информацией
        self.tab_widget = QTabWidget()

        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addWidget(self.tab_widget)

    def _setup_patients_table(self):
        """Настройка таблицы пациентов"""
        self.patients_table = QTableView()
        self.patients_model.setQuery("SELECT id, name, sex, year_of_birthday FROM pacients", self.db)

        headers = ["Ид", "Имя", "Пол", "Год рождения", "Редактировать"]
        for i, header in enumerate(headers):
            self.patients_model.setHeaderData(i, Qt.Horizontal, header)

        self.patients_table.setModel(self.patients_model)
        self.patients_table.setSelectionBehavior(QTableView.SelectRows)
        self.patients_table.clicked.connect(self._on_patient_clicked)
        self.patients_table.setColumnHidden(0, True)
        self.appointment_edit_delegate = EditPacientDelegate(':/dbicons/edit.png', self.patients_table, self._open_dialog)
        self.patients_table.setItemDelegateForColumn(self.patients_model.columnCount() - 1, self.appointment_edit_delegate)

        self.appointment_edit_delegate = EditPacientDelegate(':/dbicons/edit.png', self.patients_table, self._open_dialog)
        self.patients_table.setItemDelegateForColumn(self.patients_model.columnCount(), self.appointment_edit_delegate)
        add_button = QPushButton("Добавить пациента")
        add_button.clicked.connect(partial(self._open_dialog, 0))

        self.left_layout.addWidget(self.patients_table)
        self.left_layout.addWidget(add_button)
    def _load_patients_data(self):
        self.patients_model.setQuery("SELECT id, name, sex, year_of_birthday FROM pacients", self.db)

        # headers = ["Ид", "Имя", "Пол", "Год рождения", "Редактировать"]
        # for i, header in enumerate(headers):
        #     self.patients_model.setHeaderData(i, Qt.Horizontal, header)

        self.patients_table.setModel(self.patients_model)
        # self.patients_table.setSelectionBehavior(QTableView.SelectRows)
        # self.patients_table.clicked.connect(self._on_patient_clicked)
        # self.patients_table.setColumnHidden(0, True)

        # add_button = QPushButton("Добавить пациента")
        # add_button.clicked.connect(partial(self._open_dialog, 0))
        #
        # self.left_layout.addWidget(self.patients_table)
        # self.left_layout.addWidget(add_button)

    def _setup_appointments_table(self):
        """Настройка таблицы приемов"""
        self.appointments_table = QTableView()
        self.appointments_table.setModel(self.appointments_model)
        self.appointments_table.setSelectionBehavior(QTableView.SelectRows)
        self.appointments_table.clicked.connect(self._on_appointment_clicked)

        self.add_button_appointment = QPushButton("Добавить результаты приема")
        self.add_button_appointment.hide()
        self.add_button_appointment.clicked.connect(self._open_appointment_dialog)

        self.left_layout.addWidget(self.appointments_table)
        self.left_layout.addWidget(self.add_button_appointment)

    def _load_initial_data(self):
        """Загрузка начальных данных"""
        # Можно добавить предварительную загрузку данных при необходимости
        pass

    def _on_patient_clicked(self, index):
        """Обработчик клика по пациенту"""
        row = index.row()
        self.patient_id_clicked = self.patients_model.index(row, 0).data()
        patient_id = self.patients_model.index(row, 0).data()
        self.add_button_appointment.show()
        self._load_appointments(patient_id)
        if (self.tab_widget.count() > 0):
            self.tab_widget.clear()

    def _on_appointment_clicked(self, index):
        """Обработчик клика по приему"""
        row = index.row()
        appointment_id = self.appointments_model.index(row, 0).data()
        self._load_appointment_data(appointment_id)
        self._update_ui(appointment_id)

    def _load_appointments(self, patient_id: int):
        """Загрузка списка приемов для пациента"""
        query = QSqlQuery(self.db)
        if not query.exec(f"SELECT id, date, duration_of_the_disease FROM appointments WHERE pacient_id = {patient_id}"):
            QMessageBox.warning(self, "Ошибка", f"Ошибка загрузки приемов: {query.lastError().text()}")
            return

        self.appointments_model.setQuery(query)
        headers = ["Ид", "Дата приема", "Длит. болезни", "Редактировать", "Показать"]
        for i, header in enumerate(headers):
            self.appointments_model.setHeaderData(i, Qt.Horizontal, header)

        self.appointments_table.setColumnHidden(0, True)

    def _load_appointment_data(self, appointment_id: int):
        """Загрузка данных о приеме"""
        # query = QSqlQuery(self.db)
        # query.prepare("SELECT * FROM eyes WHERE appointment_id = ?")
        # query.addBindValue(appointment_id)
        #
        # if not query.exec():
        #     QMessageBox.warning(self, "Ошибка", f"Ошибка загрузки данных приема: {query.lastError().text()}")
        #     return
        columns = [
            "id", "eye", "appointment_id", "date", "duration_of_the_disease", "topkon", "areds", "refraction", "type_of_neovascularization",
            "choroidal_thickness_center", "cts_foveola", "total_volume", "average_volume",
            "rpe_status", "rpe_localisation", "cme_localisation",
            "serouz_rpe_detachment_localisation", "serouz_rpe_detachment_width", "serouz_rpe_detachment_height",
            "serouz_rpe_detachment_area",
            "hemorrhagic_rpe_detachment_localisation", "hemorrhagic_rpe_detachment_width",
            "hemorrhagic_rpe_detachment_heidgt", "hemorrhagic_rpe_detachment_area",
            "fibrovascular_rpe_detachment_localisation", "fibrovascular_rpe_detachment_width",
            "fibrovascular_rpe_detachment_heidgt", "fibrovascular_rpe_detachment_area",
            "drusenoid_detachment_rpe_localisation", "drusenoid_detachment_rpe_width",
            "drusenoid_detachment_rpe_height", "drusenoid_detachment_rpe_area",
            "druses_localisation", "druses_weigt", "druses_heigt", "dzuses_area",
            "fluid_under_rpe_area", "fluid_under_rpe_localisation",
            "ez_status", "ez_localisation", "myoidnz_status", "myoidnz_localisation",
            "rne_detachment_localisation", "rne_detachment_width", "rne_detachment_heigt", "rne_detachment_area",
            "hyperreflective_material_localisation", "hyperreflective_material_area"
        ]

        # Формируем SQL-запрос
        column_clause = ", ".join(columns)
        query_text = f"SELECT {column_clause} FROM eyes WHERE appointment_id = {appointment_id}"

        # Выполняем запрос
        model = QSqlQueryModel()
        query = QSqlQuery(self.db_manager.get_connection())
        query.exec_(query_text)
        model.setQuery(query)
        self.appointment_id = appointment_id
        self.appointment_data_model = model

    def _update_ui(self, eye_id_field):
        """Обновление интерфейса после загрузки данных"""

        if (hasattr(self, "editable_fields")):
            self.editable_fields.clear()
        if (self.tab_widget.count() > 0):
            self.tab_widget.clear()

        if eye_id_field is None:
            self._load_appointment_data(0)
        if hasattr(self, "formDict") and isinstance(self.formDict, DynamicFormBuilder):
            print("🧹 Очищаем форму...")
            self.formDict[0].clear_all_fields()
            self.formDict[1].clear_all_fields()
        else:
            print("⚠️ Форма не найдена или не инициализирована.")
        ids = []
        rId = None
        lId = None
        if eye_id_field is not None :
            for row in range(self.appointment_data_model.rowCount()):
                index = self.appointment_data_model.index(row, 0)  # Предположим, что id в первом столбце
                id_value = self.appointment_data_model.data(index)
                eye_in = self.appointment_data_model.data(self.appointment_data_model.index(row, 1))
                if eye_in == 'L' or eye_in == '1':
                    lId = id_value
                if eye_in == 'R' or eye_in == '0':
                    rId = id_value
                ids.append(id_value)


        # Добавление вкладок для каждого глаза
        self.tab_widget.addTab(self._create_eye_tab(0, rId), "Правый глаз")
        self.tab_widget.addTab(self._create_eye_tab(1, lId), "Левый глаз")


    def _create_eye_tab(self, eye_index: int, eye_id_field: int) -> QScrollArea:
        """Создание вкладки с данными о глазе"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content = QWidget()

        blocks = [
            ("Обследование", [
                ("Дата посещения", 3),
                ("Продолжительность заболевания", 4),
                ("Тип томографа", 5),
                ("Критерий AREDS", 6),
                ("Рефракция", 7),
                ("Тип неоваскуляризации", 8)
            ]),
            ("Ретинальные показатели", [
                ("Толщина хориоидеи в центре", 9),
                ("Толщина сетчатки в фовеоле", 10),
                ("Общий объем", 11),
                ("Средний объем", 12)
            ]),
            ("", [
                ("Состояние РПЭ", 13),
                ("Локализация дефектов РПЭ", 14),
                ("Локализация кистозного макулярного отека", 15)
            ]),
            ("Серозная ОПЭ", [
                ("Локализация", 16),
                ("Ширина", 17),
                ("Высота", 18),
                ("Площадь", 19)
            ]),
            ("Геморрагическая ОПЭ", [
                ("Локализация", 20),
                ("Ширина", 21),
                ("Высота", 22),
                ("Площадь", 23)
            ]),
            ("Фиброваскулярная ОПЭ", [
                ("Локализация", 24),
                ("Ширина", 25),
                ("Высота", 26),
                ("Площадь", 27)
            ]),
            ("Друзеноидная ОПЭ", [
                ("Локализация", 28),
                ("Ширина", 29),
                ("Высота", 30),
                ("Площадь", 31)
            ]),
            ("Друзы", [
                ("Локализация", 32),
                ("Ширина", 33),
                ("Высота", 34),
                ("Площадь", 35)
            ]),
            ("Жидкость под РПЭ", [
                ("Пощадь", 36),
                ("Локализация", 37)
            ]),
            ("Эллипсоидная зона", [
                ("Состояние", 38),
                ("Локализация дефектов", 39)
            ]),
            ("Миоидная зона", [
                ("Состояние", 40),
                ("Локализация дефектов", 41)
            ]),
            ("Отслойка нейросенсорной сетчатки", [
                ("Локализация", 42),
                ("Ширина", 43),
                ("Высота", 44),
                ("Площадь", 45)
            ]),
            ("Гиперрефлективный материал", [
                ("Локализация", 46),
                ("Площадь", 47)
            ])]
        layoutBoxes = QVBoxLayout()
        initial_data = {}
        if eye_id_field is not None:
            initial_data = self.extract_initial_data(self.appointment_data_model, eye_index)
        new_data = {}

        for key, value in initial_data.items():
            if value:  # Проверка, что строка не пустая
                new_key = str(key) + str(eye_index)
                new_data[new_key] = value
        self.formDict[eye_index] = DynamicFormBuilder(blocks, new_data, eye_index, self.appointment_data_model, self.db_manager.get_connection(), self.patient_id_clicked)
        layoutBoxes.addWidget(self.formDict[eye_index])
        # for title, fields in blocks:
        #     layoutBoxes.addWidget(self._create_data_block(title, fields, eye_index, eye_id_field))

        if eye_id_field is None:
            self.edit_toggle_button = QPushButton("Сохранить")
            self.edit_toggle_button.clicked.connect(lambda: self._toggle_edit_mode_2(self.edit_toggle_button, eye_index))
            layoutBoxes.addWidget(self.edit_toggle_button)

        content.setLayout(layoutBoxes)
        scroll_area.setWidget(content)
        return scroll_area

    def extract_initial_data(self, model, row_index):
        initial_data = {}
        column_index_to_field_name = {
            3: "date",
            4: "duration_of_the_disease",
            5: "topkon",
            6: "areds",
            7: "refraction",
            8: "type_of_neovascularization",
            9: "choroidal_thickness_center",
            10: "cts_foveola",
            11: "total_volume",
            12: "average_volume",
            13: "rpe_status",
            14: "rpe_localisation",
            15: "cme_localisation",
            16: "serouz_rpe_detachment_localisation",
            17: "serouz_rpe_detachment_width",
            18: "serouz_rpe_detachment_height",
            19: "serouz_rpe_detachment_area",
            20: "hemorrhagic_rpe_detachment_localisation",
            21: "hemorrhagic_rpe_detachment_width",
            22: "hemorrhagic_rpe_detachment_heidgt",
            23: "hemorrhagic_rpe_detachment_area",
            24: "fibrovascular_rpe_detachment_localisation",
            25: "fibrovascular_rpe_detachment_width",
            26: "fibrovascular_rpe_detachment_heidgt",
            27: "fibrovascular_rpe_detachment_area",
            28: "drusenoid_detachment_rpe_localisation",
            29: "drusenoid_detachment_rpe_width",
            30: "drusenoid_detachment_rpe_height",
            31: "drusenoid_detachment_rpe_area",
            32: "druses_localisation",
            33: "druses_weigt",
            34: "druses_heigt",
            35: "dzuses_area",
            36: "fluid_under_rpe_area",
            37: "fluid_under_rpe_localisation",
            38: "ez_status",
            39: "ez_localisation",
            40: "myoidnz_status",
            41: "myoidnz_localisation",
            42: "rne_detachment_localisation",
            43: "rne_detachment_width",
            44: "rne_detachment_heigt",
            45: "rne_detachment_area",
            46: "hyperreflective_material_localisation",
            47: "hyperreflective_material_area"
        }

        for column, field_name in column_index_to_field_name.items():
            index = model.index(row_index, column)
            if index.isValid():
                value = index.data()
                initial_data[field_name] = value
            else:
                print(f"⚠️ Недопустимый индекс: строка {row_index}, колонка {column}")
        return initial_data

    def clear_form_layout(self, form_layout5):
        while form_layout5.count():
            item = form_layout5.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _create_data_block(self, title: str, fields: list, eye_index: int, eye_id_field: int) -> QGroupBox:
        # """Создание блока с данными"""
        # group = QGroupBox(title)
        # form = QFormLayout()
        #
        # for field, column in fields:
        #     value = str(column) if isinstance(column, str) else \
        #         str(self.appointment_data_model.index(eye_index, column).data())
        #     line_edit = QLineEdit(value)
        #     line_edit.setReadOnly(True)  # Для отображения, не редактирования
        #     form.addRow(f"{field}:", line_edit)
        #
        # group.setLayout(form)
        # return group
        """Создание блока с возможностью редактирования и сохранения"""
        group = QGroupBox(title)
        layout = QVBoxLayout()
        form = QFormLayout()

        # Храним поля и кнопку
        self.editable_fields = getattr(self, "editable_fields", {})
        self.editable_fields[group] = []
        #
        # for field, column in fields:
        #     value = '' if eye_id_field is None else str(column) if isinstance(column, str) else \
        #         str(self.appointment_data_model.index(eye_index, column).data())
        #     line_edit = QLineEdit(value)
        #     if eye_id_field is None:
        #         line_edit.setReadOnly(False)
        #         line_edit.setStyleSheet("background-color: #ffffff;")# Стиль для readonly
        #     else:
        #         line_edit.setReadOnly(True)
        #         line_edit.setStyleSheet("background-color: #f0f0f0;")  # Стиль для readonly
        #     form.addRow(f"{field}:", line_edit)
        #     self.editable_fields[group].append((line_edit, column))  # сохраняем и индекс
        for label, field_name in self.editable_fields:
            value = self.appointment_data_model.index(eye_index, field_name).data() # значение из модели или базы
            widget = self.create_field_widget(field_name, value)
            layout.addRow(label, widget)

            self.field_widgets[field_name] = widget

        layout.addLayout(form)

        if eye_id_field is not None:
        # Кнопка редактирования/сохранения
            edit_button = QPushButton("Редактировать")
            edit_button.clicked.connect(lambda _, g=group, b=edit_button, i=eye_index: self._toggle_edit_mode(g, b, i, eye_id_field))
            layout.addWidget(edit_button)

        group.setLayout(layout)
        return group

    def get_field_value(self, widget):
        if isinstance(widget, QComboBox):
            return widget.currentData()  # значение для базы
        elif isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def create_field_widget(self, field_name, initial_value=None):
        # Если поле имеет предопределённые значения — создаём QComboBox
        if field_name in self.dropdown_db_values:
            combo = QComboBox()
            db_values = self.dropdown_db_values[field_name]
            display_map = self.dropdown_display_map[field_name]

            # Добавляем отображаемые значения
            for db_value in db_values:
                display_text = display_map.get(db_value, db_value)
                combo.addItem(display_text, db_value)

            # Устанавливаем начальное значение, если есть
            if initial_value in db_values:
                index = db_values.index(initial_value)
                combo.setCurrentIndex(index)

            return combo
        else:
            # Обычное текстовое поле
            line_edit = QLineEdit()
            if initial_value is not None:
                line_edit.setText(str(initial_value))
            return line_edit


    def _toggle_edit_mode(self, group: QGroupBox, button: QPushButton, eye_index: int, eye_id_field: int):
        """Переключает режим редактирования и сохраняет данные"""
        is_editing = button.text() == "Редактировать"

        for line_edit, column in self.editable_fields.get(group, []):
            line_edit.setReadOnly(not is_editing)
            if is_editing:
                line_edit.setStyleSheet("background-color: #ffffff;")  # активный стиль
            else:
                line_edit.setStyleSheet("background-color: #f0f0f0;")  # обратно

        if is_editing:
            button.setText("Сохранить")
        else:
            button.setText("Редактировать")
            self._save_block_data(group, eye_index, eye_id_field)  # вызов сохранения

    def _toggle_edit_mode_2(self, button: QPushButton, eye_index: int):
        """Переключает режим редактирования всех блоков и вставляет новую запись"""
        is_editing = button.text() == "Редактировать"

        # Переключаем все поля
        table_name = "eyes"
        all_fields = []
        all_values = {}
        blocks = [
            ("Обследование", [
                ("Дата посещения", 3),
                ("Продолжительность заболевания", 4),
                ("Тип томографа", 5),
                ("Критерий AREDS", 6),
                ("Рефракция", 7),
                ("Тип неоваскуляризации", 8)
            ]),
            ("Ретинальные показатели", [
                ("Толщина хориоидеи в центре", 9),
                ("Толщина сетчатки в фовеоле", 10),
                ("Общий объем", 11),
                ("Средний объем", 12)
            ]),
            ("", [
                ("Состояние РПЭ", 13),
                ("Локализация дефектов РПЭ", 14),
                ("Локализация кистозного макулярного отека", 15)
            ]),
            ("Серозная ОПЭ", [
                ("Локализация", 16),
                ("Ширина", 17),
                ("Высота", 18),
                ("Площадь", 19)
            ]),
            ("Геморрагическая ОПЭ", [
                ("Локализация", 20),
                ("Ширина", 21),
                ("Высота", 22),
                ("Площадь", 23)
            ]),
            ("Фиброваскулярная ОПЭ", [
                ("Локализация", 24),
                ("Ширина", 25),
                ("Высота", 26),
                ("Площадь", 27)
            ]),
            ("Друзеноидная ОПЭ", [
                ("Локализация", 28),
                ("Ширина", 29),
                ("Высота", 30),
                ("Площадь", 31)
            ]),
            ("Друзы", [
                ("Локализация", 32),
                ("Ширина", 33),
                ("Высота", 34),
                ("Площадь", 35)
            ]),
            ("Жидкость под РПЭ", [
                ("Пощадь", 36),
                ("Локализация", 37)
            ]),
            ("Эллипсоидная зона", [
                ("Состояние", 38),
                ("Локализация дефектов", 39)
            ]),
            ("Миоидная зона", [
                ("Состояние", 40),
                ("Локализация дефектов", 41)
            ]),
            ("Отслойка нейросенсорной сетчатки", [
                ("Локализация", 42),
                ("Ширина", 43),
                ("Высота", 44),
                ("Площадь", 45)
            ]),
            ("Гиперрефлективный материал", [
                ("Локализация", 46),
                ("Площадь", 47)
            ])]


        # Сбор данных из всех виджетов
        for block_title, fields in blocks:
            for label, field_name2 in fields:
                field_name = self._get_column_name(field_name2)
                widget = self.formDict[eye_index].get_all_field_widgets().get(field_name + str(eye_index))
                if widget is None:
                    continue

                # Получение значения
                if isinstance(widget, QComboBox):
                    value = widget.currentData()
                elif isinstance(widget, QLineEdit):
                    value = widget.text()
                else:
                    continue

                all_fields.append(field_name)
                if value is not None and str(value).strip():
                    all_values[field_name] = value

                print(f"{field_name} = '{value}'")  # отладка

        # Проверка на наличие данных
        if not all_fields:
            print("❌ Нет данных для вставки.")
            return

        if (self.appointment_id is None):
            # Вставка в appointments
            query1 = QSqlQuery(self.db_manager.get_connection())
            query1.prepare("""
                               INSERT INTO appointments (pacient_id, date, duration_of_the_disease)
                               VALUES (:pacient_id, :date2, :duration_of_the_disease2)
                           """)
            query1.bindValue(":pacient_id", self.patient_id_clicked)
            query1.bindValue(":date2", all_values.get("date"))
            query1.bindValue(":duration_of_the_disease2", all_values.get("duration_of_the_disease"))

            if not query1.exec_():
                print("❌ Ошибка при вставке appointments:", query1.lastError().text())
                return
            self.appointment_id = query1.lastInsertId()
            print(f"✅ Создана запись appointments с ID: {self.appointment_id}")

        # Добавляем служебные поля
        all_fields += ["appointment_id", "eye", "MKO", "topkon", "optopol"]
        all_values["appointment_id"] = self.appointment_id
        all_values["eye"] = eye_index
        all_values["MKO"] = eye_index
        all_values["topkon"] = True
        all_values["optopol"] = True

        # Формируем SQL-запрос
        field_clause = ", ".join(all_fields)
        placeholder_clause = ", ".join([f":{field}" for field in all_fields])
        query_text = f"INSERT INTO {table_name} ({field_clause}) VALUES ({placeholder_clause})"

        query2 = QSqlQuery(self.db_manager.get_connection())
        query2.prepare(query_text)

        for field, value in all_values.items():
            query2.bindValue(f":{field}", value)

        if not query2.exec_():
            print("❌ Ошибка при вставке eyes:", query2.lastError().text())
        else:
            print("✅ Запись в eyes успешно добавлена.")
            self._load_appointments(self.patient_id_clicked)

    def insert_new_eye_record(self, eye_index, patient_id_clicked, appointment_data_model, db_manager):
        table_name = "eyes"
        all_fields = []
        all_values = {}
        blocks = [
            ("Обследование", [
                ("Дата посещения", 3),
                ("Продолжительность заболевания", 4),
                ("Тип томографа", 5),
                ("Критерий AREDS", 6),
                ("Рефракция", 7),
                ("Тип неоваскуляризации", 8)
            ]),
            ("Ретинальные показатели", [
                ("Толщина хориоидеи в центре", 9),
                ("Толщина сетчатки в фовеоле", 10),
                ("Общий объем", 11),
                ("Средний объем", 12)
            ]),
            ("", [
                ("Состояние РПЭ", 13),
                ("Локализация дефектов РПЭ", 14),
                ("Локализация кистозного макулярного отека", 15)
            ]),
            ("Серозная ОПЭ", [
                ("Локализация", 16),
                ("Ширина", 17),
                ("Высота", 18),
                ("Площадь", 19)
            ]),
            ("Геморрагическая ОПЭ", [
                ("Локализация", 20),
                ("Ширина", 21),
                ("Высота", 22),
                ("Площадь", 23)
            ]),
            ("Фиброваскулярная ОПЭ", [
                ("Локализация", 24),
                ("Ширина", 25),
                ("Высота", 26),
                ("Площадь", 27)
            ]),
            ("Друзеноидная ОПЭ", [
                ("Локализация", 28),
                ("Ширина", 29),
                ("Высота", 30),
                ("Площадь", 31)
            ]),
            ("Друзы", [
                ("Локализация", 32),
                ("Ширина", 33),
                ("Высота", 34),
                ("Площадь", 35)
            ]),
            ("Жидкость под РПЭ", [
                ("Пощадь", 36),
                ("Локализация", 37)
            ]),
            ("Эллипсоидная зона", [
                ("Состояние", 38),
                ("Локализация дефектов", 39)
            ]),
            ("Миоидная зона", [
                ("Состояние", 40),
                ("Локализация дефектов", 41)
            ]),
            ("Отслойка нейросенсорной сетчатки", [
                ("Локализация", 42),
                ("Ширина", 43),
                ("Высота", 44),
                ("Площадь", 45)
            ]),
            ("Гиперрефлективный материал", [
                ("Локализация", 46),
                ("Площадь", 47)
            ])]

        # Сбор данных из всех виджетов
        for block_title, fields in blocks:
            for label, field_name in fields:
                widget = self.formDict[eye_index].get_all_field_widgets().get(field_name + str(eye_index))
                if widget is None:
                    continue

                # Получение значения
                if isinstance(widget, QComboBox):
                    value = widget.currentData()
                elif isinstance(widget, QLineEdit):
                    value = widget.text()
                else:
                    continue

                all_fields.append(field_name)
                if value is not None and str(value).strip():
                    all_values[field_name] = value

                print(f"{field_name} = '{value}'")  # отладка

        # Проверка на наличие данных
        if not all_fields:
            print("❌ Нет данных для вставки.")
            return

        # Вставка в appointments
        query1 = QSqlQuery(db_manager.get_connection())
        query1.prepare("""
            INSERT INTO appointments (pacient_id, date, duration_of_the_disease)
            VALUES (:pacient_id, :date2, :duration_of_the_disease2)
        """)
        query1.bindValue(":pacient_id", patient_id_clicked)
        query1.bindValue(":date2", all_values.get("date"))
        query1.bindValue(":duration_of_the_disease2", all_values.get("duration_of_the_disease"))

        if not query1.exec_():
            print("❌ Ошибка при вставке appointments:", query1.lastError().text())
            return

        appointment_id = query1.lastInsertId()
        print(f"✅ Создана запись appointments с ID: {appointment_id}")

        # Добавляем служебные поля
        all_fields += ["appointment_id", "eye", "MKO", "topkon", "optopol"]
        all_values["appointment_id"] = appointment_id
        all_values["eye"] = eye_index
        all_values["MKO"] = eye_index
        all_values["topkon"] = True
        all_values["optopol"] = True

        # Формируем SQL-запрос
        field_clause = ", ".join(all_fields)
        placeholder_clause = ", ".join([f":{field}" for field in all_fields])
        query_text = f"INSERT INTO {table_name} ({field_clause}) VALUES ({placeholder_clause})"

        query2 = QSqlQuery(db_manager.get_connection())
        query2.prepare(query_text)

        for field, value in all_values.items():
            query2.bindValue(f":{field}", value)

        if not query2.exec_():
            print("❌ Ошибка при вставке eyes:", query2.lastError().text())
        else:
            print("✅ Запись в eyes успешно добавлена.")
            self._load_appointments(patient_id_clicked)

    def _insert_new_record(self, eye_index: int):
        """Собирает данные из всех блоков и вставляет новую запись в базу"""
        # table_name = "eyes"
        # all_fields = []
        # all_values = {}
        #
        # for group, field_list in self.editable_fields.items():
        #     for line_edit, column in field_list:
        #         if isinstance(column, int):
        #             field_name = self._get_column_name(column)
        #             value = line_edit.text()
        #             all_fields.append(field_name)
        #             all_values[field_name] = value
        #
        # if not all_fields:
        #     print("❌ Нет данных для вставки.")
        #     return
        #
        # field_clause = ", ".join(all_fields)
        # placeholder_clause = ", ".join([f":{field}" for field in all_fields])
        # query_text = (f"INSERT INTO appointments (pacient_id, date, duration_of_the_disease) VALUES (:pacient_id, :date2, :duration_of_the_disease2);"
        #               f" INSERT INTO {table_name} ({field_clause}) VALUES ({placeholder_clause})")
        #
        # query = QSqlQuery(self.db_manager.get_connection())
        # query.prepare(query_text)
        #
        # for field, value in all_values.items():
        #     query.bindValue(f":{field}", value)
        # query.bindValue(":pacient_id", self.patient_id_clicked)
        # query.bindValue(":date2", all_values.get("date"))
        # query.bindValue(":duration_of_the_disease2", all_values.get("duration_of_the_disease"))
        #
        # print("Запрос:", query.lastQuery())
        # print("Параметры:")
        # for field, value in all_values.items():
        #     print(field, query.boundValue(field))
        #
        # if not query.exec_():
        #     print(f"❌ Ошибка при вставке: {query.lastError().text()}")
        # else:
        #     print("✅ Новая запись успешно добавлена.")
        table_name = "eyes"
        all_fields = []
        all_values = {}

        for group, field_list in self.editable_fields.items():
            for line_edit, column in field_list:
                if isinstance(column, int):
                    field_name = self._get_column_name(column)
                    if field_name is not None:
                        value = line_edit.text()
                        all_fields.append(field_name)
                        if value is not None and str(value).strip():
                            all_values[field_name] = value
                        print(f"{field_name} = '{value}'")  # отладка

        all_values['eye'] = eye_index
        all_values['МКОЗ'] = eye_index
        # Проверка на наличие данных
        if not all_fields:
            print("❌ Нет данных для вставки.")
            return

        # Вставка в appointments
        query1 = QSqlQuery(self.db_manager.get_connection())
        query1.prepare("""
            INSERT INTO appointments (pacient_id, date, duration_of_the_disease)
            VALUES (:pacient_id, :date2, :duration_of_the_disease2)
        """)
        query1.bindValue(":pacient_id", self.patient_id_clicked)
        query1.bindValue(":date2", all_values.get("date"))
        query1.bindValue(":duration_of_the_disease2", all_values.get("duration_of_the_disease"))

        if not query1.exec_():
            print("❌ Ошибка при вставке appointments:", query1.lastError().text())
            return

        # Получаем ID созданной записи
        appointment_id = query1.lastInsertId()
        print(f"✅ Создана запись appointments с ID: {appointment_id}")

        # Добавляем appointment_id в данные для eyes
        all_fields.append("appointment_id")
        all_fields.append("eye")
        all_fields.append("MKO")
        all_fields.append("topkon")
        all_fields.append("optopol")
        all_values["appointment_id"] = appointment_id
        all_values['eye'] = eye_index
        all_values['MKO'] = eye_index
        all_values['topkon'] = True
        all_values['optopol'] = True

        # Формируем SQL-запрос для eyes
        field_clause = ", ".join(all_fields)
        placeholder_clause = ", ".join([f":{field}" for field in all_fields])
        query_text = f"INSERT INTO {table_name} ({field_clause}) VALUES ({placeholder_clause})"

        query2 = QSqlQuery(self.db_manager.get_connection())
        query2.prepare(query_text)

        for field, value in all_values.items():
            query2.bindValue(f":{field}", value)

        if not query2.exec_():
            print("❌ Ошибка при вставке eyes:", query2.lastError().text())
        else:
            print("✅ Запись в eyes успешно добавлена.")
            self._load_appointments(self.patient_id_clicked)

    def _get_column_name(self, column_index: int) -> str:
        """Возвращает имя поля по индексу"""
        # column_index_to_field_name = {
        #     3: "date",  # Дата посещения
        #     4: "duration_of_the_disease",  # Продолжительность заболевания
        #     5: "topkon",  # Тип томографа — "Топкон"
        #     7: "areds",  # Критерий AREDS
        #     8: "refraction",  # Рефракция
        #     9: "type_of_neovascularization",  # Тип неоваскуляризации
        #     10: "choroidal_thickness_center",  # Толщина хориоидеи в центре
        #     11: "cts_foveola",  # Толщина сетчатки в фовеоле
        #     14: "total_volume",  # Общий объем
        #     15: "average_volume",  # Средний объем
        #     16: "rpe_status",  # Состояние РПЭ
        #     17: "rpe_localisation",  # Локализация дефектов РПЭ
        #     18: "cme_localisation",  # Локализация кистозного макулярного отека
        #     19: "serouz_rpe_detachment_localisation",  # Серозная ОПЭ — Локализация
        #     20: "serouz_rpe_detachment_width",  # Ширина
        #     21: "serouz_rpe_detachment_height",  # Высота
        #     22: "serouz_rpe_detachment_area",  # Площадь
        #     23: "hemorrhagic_rpe_detachment_localisation",  # Геморрагическая ОПЭ — Локализация
        #     24: "hemorrhagic_rpe_detachment_width",  # Ширина
        #     25: "hemorrhagic_rpe_detachment_heidgt",  # Высота (опечатка: должно быть height)
        #     26: "hemorrhagic_rpe_detachment_area",  # Площадь
        #     27: "fibrovascular_rpe_detachment_localisation",  # Фиброваскулярная ОПЭ — Локализация
        #     28: "fibrovascular_rpe_detachment_width",  # Ширина
        #     29: "fibrovascular_rpe_detachment_heidgt",  # Высота (опечатка: должно быть height)
        #     30: "fibrovascular_rpe_detachment_area",  # Площадь
        #     31: "drusenoid_detachment_rpe_localisation",  # Друзеноидная ОПЭ — Локализация
        #     32: "drusenoid_detachment_rpe_width",  # Ширина
        #     33: "drusenoid_detachment_rpe_height",  # Высота
        #     34: "drusenoid_detachment_rpe_area",  # Площадь
        #     35: "druses_localisation",  # Друзы — Локализация
        #     36: "druses_weigt",  # Ширина (опечатка: должно быть weight)
        #     37: "druses_heigt",  # Высота (опечатка: должно быть height)
        #     38: "dzuses_area",  # Площадь (опечатка: должно быть druses_area)
        #     39: "fluid_under_rpe_area",  # Жидкость под РПЭ — Площадь
        #     40: "fluid_under_rpe_localisation",  # Локализация
        #     41: "ez_status",  # Эллипсоидная зона — Состояние
        #     42: "ez_localisation",  # Локализация дефектов
        #     43: "myoidnz_status",  # Миоидная зона — Состояние
        #     44: "myoidnz_localisation",  # Локализация дефектов
        #     45: "rne_detachment_localisation",  # Отслойка нейросенсорной сетчатки — Локализация
        #     46: "rne_detachment_width",  # Ширина
        #     47: "rne_detachment_heigt",  # Высота (опечатка: должно быть height)
        #     48: "rne_detachment_area",  # Площадь
        #     49: "hyperreflective_material_localisation",  # Гиперрефлективный материал — Локализация
        #     50: "hyperreflective_material_area"  # Площадь
        # }
        column_index_to_field_name = {
            3: "date",
            4: "duration_of_the_disease",
            5: "topkon",
            6: "areds",
            7: "refraction",
            8: "type_of_neovascularization",
            9: "choroidal_thickness_center",
            10: "cts_foveola",
            11: "total_volume",
            12: "average_volume",
            13: "rpe_status",
            14: "rpe_localisation",
            15: "cme_localisation",
            16: "serouz_rpe_detachment_localisation",
            17: "serouz_rpe_detachment_width",
            18: "serouz_rpe_detachment_height",
            19: "serouz_rpe_detachment_area",
            20: "hemorrhagic_rpe_detachment_localisation",
            21: "hemorrhagic_rpe_detachment_width",
            22: "hemorrhagic_rpe_detachment_heidgt",
            23: "hemorrhagic_rpe_detachment_area",
            24: "fibrovascular_rpe_detachment_localisation",
            25: "fibrovascular_rpe_detachment_width",
            26: "fibrovascular_rpe_detachment_heidgt",
            27: "fibrovascular_rpe_detachment_area",
            28: "drusenoid_detachment_rpe_localisation",
            29: "drusenoid_detachment_rpe_width",
            30: "drusenoid_detachment_rpe_height",
            31: "drusenoid_detachment_rpe_area",
            32: "druses_localisation",
            33: "druses_weigt",
            34: "druses_heigt",
            35: "dzuses_area",
            36: "fluid_under_rpe_area",
            37: "fluid_under_rpe_localisation",
            38: "ez_status",
            39: "ez_localisation",
            40: "myoidnz_status",
            41: "myoidnz_localisation",
            42: "rne_detachment_localisation",
            43: "rne_detachment_width",
            44: "rne_detachment_heigt",
            45: "rne_detachment_area",
            46: "hyperreflective_material_localisation",
            47: "hyperreflective_material_area"
        }

        # maybe_name = self.appointment_data_model.headerData(column_index, Qt.Horizontal) надо разобраться
        return column_index_to_field_name.get(column_index)

    def get_column_index_by_name(self, model, column_name: str) -> int:
        """Возвращает индекс столбца по его названию"""
        for col in range(model.columnCount()):
            header = model.headerData(col, Qt.Horizontal)
            if header == column_name:
                return col
        return -1  # если не найдено

    def _save_block_data(self, group: QGroupBox, eye_index: int, eye_id_field: int):
        table_name = "eyes"
        fields = []
        values = {}

        # Сбор данных из формы
        for line_edit, column in self.editable_fields.get(group, []):
            if isinstance(column, int):
                field_name = self._get_column_name(column)

                value = line_edit.text()
                if value is not None and str(value).strip():
                    fields.append(field_name)
                    values[field_name] = value

                    # Обновление модели (если редактируемая)
                    index = self.appointment_data_model.index(eye_index, column)
                    if index.isValid():
                        self.appointment_data_model.setData(index, value)

        # Подготовка SQL-запроса
        eye_id_name = 'id'
        eye_id_value = self.appointment_data_model.index(eye_index, 0).data()

        if fields and values and eye_id_value is not None:
            set_clause = ", ".join([f"{field} = :{field}" for field in fields])
            query_text = f"""
                UPDATE {table_name}
                SET {set_clause}
                WHERE {eye_id_name} = :{eye_id_name}
            """

            query = QSqlQuery(self.db_manager.get_connection())
            query.prepare(query_text)

            # Привязка значений
            for field, value in values.items():
                field_clean = field
                column_index = self.get_column_index_by_name(self.appointment_data_model, field_clean)

                index = self.appointment_data_model.index(eye_index, column_index)
                raw_data = index.data() if index.isValid() else None
                data_type = type(raw_data) if raw_data is not None else str

                try:
                    converted_value = data_type(value) if callable(data_type) else value
                except Exception as e:
                    print(f"❌ Ошибка преобразования поля {field_clean}: {e}")
                    converted_value = value

                query.bindValue(f":{field_clean}", converted_value)

            query.bindValue(f":{eye_id_name}", eye_id_value)

            # Отладка
            print("📤 SQL-запрос:")
            print(query_text)
            print("📦 Параметры:")
            for field in values:
                print(f"{field} → {query.boundValue(f':{field}')}")
                print(f"{eye_id_name} → {eye_id_value}")

                # Выполнение
            if not query.exec_():
                print(f"❌ Ошибка при обновлении: {query.lastError().text()}")
            else:
                print(f"✅ Данные блока '{group.title()}' успешно обновлены.")
        # """Обновляет данные из блока в базе данных через QSqlDatabase"""
        # table_name = "eyes"  # Название таблицы
        # fields = []
        # values = {}
        #
        # for i, (line_edit, column) in enumerate(self.editable_fields.get(group, [])):
        #     if isinstance(column, int):
        #         field_name2 = self._get_column_name(column)
        #         field_name = '"МКОЗ"' if field_name2 == 'МКОЗ' else field_name2
        #         value = line_edit.text()
        #         fields.append(field_name)
        #         values[field_name] = value
        #
        #         # Обновление модели
        #         index = self.appointment_data_model.index(eye_id_field, column)
        #         self.appointment_data_model.setData(index, value)
        #
        # # Добавляем идентификатор записи (например, eye_id)
        # eye_id_value = eye_index  # или другой уникальный ID, если есть
        # eye_id_name = 'id'
        #
        # if fields and values:
        #     set_clause = ", ".join([f"{field} = :{field}" for field in fields])
        #     query_text = f"""
        #             UPDATE {table_name}
        #             SET {set_clause}
        #             WHERE {eye_id_name} = :{eye_id_name}
        #         """
        #
        #     query = QSqlQuery(self.db_manager.get_connection())
        #     query.prepare(query_text)
        #
        #
        #     # Привязка значений
        #     for field, value in values.items():
        #         field2 = 'МКОЗ' if field == '"МКОЗ"' else field
        #         column_index = self.get_column_index_by_name(self.appointment_data_model, field2)
        #         if column_index != -1:
        #             print(f"Индекс столбца {field2}: {column_index}")
        #         else:
        #             print("❌ Столбец 'МКОЗ' не найден")
        #
        #         index = self.appointment_data_model.index(eye_index, column_index)
        #         data_type = type(index.data())
        #         if callable(data_type):
        #             converted_value = data_type(value)
        #             query.bindValue(f":{field2}", converted_value)
        #
        #     query.bindValue(f":{eye_id_name}", self.appointment_data_model.index(eye_index, 0).data())
        #
        #     print("Запрос:", query.lastQuery())
        #     print("Параметры:")
        #     for field, value in values.items():
        #         print(field, query.boundValue(field))
        #
        #     if not query.exec_():
        #         print(f"❌ Ошибка при обновлении: {query.lastError().text()}")
        #     else:
        #         print(f"✅ Данные блока '{group.title()}' успешно обновлены.")
        # """Сохраняет данные из блока в модель и базу данных через QSqlDatabase"""
        # table_name = "eyes"  # Название таблицы в БД
        # fields = []
        # values = []
        #
        # for line_edit, column in self.editable_fields.get(group, []):
        #     if isinstance(column, int):
        #         field_name = self._get_column_name(column)
        #         value = line_edit.text()
        #         fields.append(field_name)
        #         values.append(value)
        #
        #         # Обновление модели
        #         index = self.appointment_data_model.index(eye_index, column)
        #         self.appointment_data_model.setData(index, value)
        #
        # if fields and values:
        #     field_list = ", ".join(fields)
        #     placeholder_list = ", ".join([f":{field}" for field in fields])
        #     query_text = f"INSERT INTO {table_name} ({field_list}) VALUES ({placeholder_list})"
        #
        #     query = QSqlQuery(self.db_manager.get_connection())
        #     query.prepare(query_text)
        #
        #     for field, value in zip(fields, values):
        #         query.bindValue(f":{field}", value)
        #
        #     if not query.exec_():
        #         print(f"❌ Ошибка при сохранении: {query.lastError().text()}")
        #     else:
        #         print("✅ Данные успешно сохранены в базу.")

    def _execute_insert(self, query: str, values: list):
        """Выполняет SQL INSERT в базу данных"""
        conn = self.db_manager.get_connection()  # путь к БД
        cursor = conn.cursor()
        try:
            cursor.execute(query, values)
            conn.commit()
            print("✅ Данные успешно сохранены.")
        except Exception as e:
            print(f"❌ Ошибка при сохранении: {e}")
        finally:
            conn.close()

    def _get_column_name_app(self, column_index: int) -> str:
        """Возвращает имя поля по индексу"""
        return self.appointment_data_model.headerData(column_index, Qt.Horizontal)

    def _open_dialog(self, patient_id: int):
        """Открытие диалога добавления/редактирования пациента"""
        dialog = AddRecordDialog(self.db, patient_id)
        if dialog.exec():
            self.patients_model.setQuery("SELECT id, name, sex, year_of_birthday FROM pacients", self.db)
        self._load_patients_data()

    def _open_appointment_dialog(self):
        """Открытие диалога добавления приема"""
        """Обработчик клика по приему"""
        self.appointment_id = None
        self._update_ui(None)
        pass

    def closeEvent(self, event):
        """Обработчик закрытия окна"""
        self.db_manager.close()
        super().closeEvent(event)


# from functools import partial
#
# from PySide6.QtCore import QResource
# from PySide6.QtGui import QIcon, Qt, QPixmap
# from PySide6.QtSql import QSqlQueryModel, QSqlDatabase, QSqlQuery
# from PySide6.QtWidgets import (
#     QWidget, QTableWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QLineEdit, QLabel, QScrollArea, QGridLayout,
#     QTableView,
#     QStyledItemDelegate, QMessageBox, QGroupBox, QFormLayout, QTabWidget, QDialog
# )
#
# from bsmu.macula.plugins.db.add_patient_dialog import AddRecordDialog
# from bsmu.macula.plugins.db.edit_pacirnt_delegate import EditPacientDelegate
# from bsmu.macula.widgets.eye_data_widget import EyeDataWidget
# from bsmu.macula.records.eye_info_data import PatientExamData
# from bsmu.macula.plugins.db.images import dbicons_rc  # noqa: F401
#
#
# class TableWidgetExample(QWidget):
#
#     def __init__(self):
#         super().__init__()
#         self.init_db()
#         self.createMainTables()
#         self.setWindowTitle("Пациенты")
#         self.resize(1200, 800)
#         self.appointments_table_created = False
#         self.appointment_data_table_created = False
#
#     def init_db(self):
#         db = QSqlDatabase.addDatabase("QSQLITE")
#         db.setDatabaseName("database.db")
#         if not db.open():
#             print("Ошибка подключения к базе данных")
#             return None
#         self.db = db
#
#     def patient_clicked(self, index):
#         row = index.row()
#         user_id = self.patients_model.index(row, 0).data()
#         self.load_appointments_from_db(user_id)
#
#     def apointment_clicked(self, index):
#         row = index.row()
#         apointment_id = self.appointments_model.index(row, 0).data()
#         self.load_appointment_data_from_db(apointment_id)
#
#         self.init_ui()
#         # self.main_lay.addLayout(self.init_ui())
#
#     def createMainTables(self):
#         self.setWindowTitle("Пациенты")
#         self.resize(1200, 800)
#
#         self.main_lay = QHBoxLayout(self)
#         left_layout = QVBoxLayout(self)
#
#         db = QSqlDatabase.addDatabase("QSQLITE")
#         db.setDatabaseName("database.db")
#         if not db.open():
#             print("Ошибка подключения к базе данных")
#             return None
#
#         self.patients_model = QSqlQueryModel()
#         self.patients_model.setQuery("SELECT id, name, sex, year_of_birthday FROM pacients")
#
#         self.patients_table = QTableView(self)
#         self.patients_table.setModel(self.patients_model)
#
#         self.patients_model.insertColumn(self.patients_model.columnCount())
#
#         self.patients_model.setHeaderData(0, Qt.Orientation.Horizontal, "Ид")
#         self.patients_model.setHeaderData(1, Qt.Orientation.Horizontal, "Имя")
#         self.patients_model.setHeaderData(2, Qt.Orientation.Horizontal, "Пол")
#         self.patients_model.setHeaderData(3, Qt.Orientation.Horizontal, "Год рождения")
#         self.patients_model.setHeaderData(4, Qt.Orientation.Horizontal, "Редактировать")
#
#         self.patients_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
#         self.patients_table.clicked.connect(self.patient_clicked)
#
#         self.patients_model.setHeaderData(4, Qt.Orientation.Horizontal, "Редактировать")
#         self.appointment_edit_delegate = EditPacientDelegate(':/dbicons/edit.png', self.patients_table, self.open_dialog)
#         self.patients_table.setItemDelegateForColumn(self.patients_model.columnCount() - 1, self.appointment_edit_delegate)
#
#
#         self.patients_table.setColumnHidden(0, True)
#
#         add_patient = QPushButton("Добавить пациента")
#         add_patient.clicked.connect(partial(self.open_dialog, 0))
#         button_layout = QHBoxLayout()
#         button_layout.addWidget(add_patient)
#
#         self.appointments_table = QTableView(self)
#         add_appointment_button = QPushButton("Добавить результаты приема")
#         button_layout2 = QHBoxLayout()
#         button_layout2.addWidget(add_appointment_button)
#
#         left_layout.addWidget(self.patients_table)
#         left_layout.addLayout(button_layout)
#         left_layout.addWidget(self.appointments_table)
#         left_layout.addLayout(button_layout2)
#         self.main_lay.addLayout(left_layout)
#         layout = QVBoxLayout()
#         self.tab_widget = QTabWidget()
#         layout.addWidget(self.tab_widget)
#         self.main_lay.addLayout(layout)
#
#     def open_full_image(self, event):
#         """Открывает изображение в отдельном окне."""
#         self.dialog.exec()
#
#     def init_ui(self):
#         if (self.tab_widget.count() > 0) :
#             self.tab_widget.removeTab(0)
#             self.tab_widget.removeTab(0)
#         self.tab_widget.addTab(self.create_scrollable_area("Правый глаз", 0), "Правый глаз")
#         self.tab_widget.addTab(self.create_scrollable_area("Левый глаз", 1), "Левый глаз")
#         eye_data = PatientExamData.from_query_model(self.appointment_data_model, 1)
#         eye_widget = EyeDataWidget(eye_data)
#         self.tab_widget.addTab(eye_widget, "Левый глаз")
#
#
#
#     def create_scrollable_area(self, layout_name, index_eye):
#         """Создаёт область прокрутки с блоками"""
#         scroll_area = QScrollArea()
#         scroll_area.setWidgetResizable(True)
#
#         # Виджет-контейнер для полей
#         scroll_content = QWidget()
#         scroll_layout = QVBoxLayout()
#
#         # Добавляем блоки в прокручиваемую область
#         scroll_layout.addWidget(self.create_block("Обследование", [
#             ("Дата посещения", self.get_app_data_in(index_eye, 3)),
#             ("Продолжительность заболевания", self.get_app_data_in(index_eye, 4)),
#             ("Тип томографа ","Топкон"),
#             ("Критерий AREDS", self.get_app_data_in(index_eye, 7)),
#             ("Рефракция", self.get_app_data_in(index_eye, 8)),
#             ("Тип неоваскуляризации", self.get_app_data_in(index_eye, 9))
#         ]))
#         scroll_layout.addWidget(self.create_block("Ретинальные показатели", [
#             ("Толщина хориоидеи в центре", self.get_app_data_in(index_eye, 10)),
#             ("Толщина сетчатки в фовеоле", self.get_app_data_in(index_eye, 11)),
#             ("Общий объем", self.get_app_data_in(index_eye, 14)),
#             ("Средний объем", self.get_app_data_in(index_eye, 15))
#         ]))
#         scroll_layout.addWidget(self.create_block("", [
#             ("Состояние РПЭ", self.get_app_data_in(index_eye, 16)),
#             ("Локализация дефектов РПЭ", self.get_app_data_in(index_eye, 17)),
#             ("Локализация кистозного макулярного отека", self.get_app_data_in(index_eye, 18))
#         ]))
#         scroll_layout.addWidget(QLabel("Отслойки РПЭ"))
#         scroll_layout.addWidget(self.create_block("Серозная ОПЭ", [
#             ("Локализация", self.get_app_data_in(index_eye, 19)),
#             ("Ширина", self.get_app_data_in(index_eye, 20)),
#             ("Высота", self.get_app_data_in(index_eye, 21)),
#             ("Площадь", self.get_app_data_in(index_eye, 22))
#         ]))
#         scroll_layout.addWidget(self.create_block("Геморрагическая ОПЭ", [
#             ("Локализация", self.get_app_data_in(index_eye, 23)),
#             ("Ширина", self.get_app_data_in(index_eye, 24)),
#             ("Высота", self.get_app_data_in(index_eye, 25)),
#             ("Площадь", self.get_app_data_in(index_eye, 26))
#         ]))
#         scroll_layout.addWidget(self.create_block("Фиброваскулярная ОПЭ", [
#             ("Локализация", self.get_app_data_in(index_eye, 27)),
#             ("Ширина", self.get_app_data_in(index_eye, 28)),
#             ("Высота", self.get_app_data_in(index_eye, 29)),
#             ("Площадь", self.get_app_data_in(index_eye, 30))
#         ]))
#         scroll_layout.addWidget(self.create_block("Друзеноидная ОПЭ", [
#             ("Локализация", self.get_app_data_in(index_eye, 31)),
#             ("Ширина", self.get_app_data_in(index_eye, 32)),
#             ("Высота", self.get_app_data_in(index_eye, 33)),
#             ("Площадь", self.get_app_data_in(index_eye, 34))
#         ]))
#         scroll_layout.addWidget(self.create_block("Друзы", [
#             ("Локализация", self.get_app_data_in(index_eye, 35)),
#             ("Ширина", self.get_app_data_in(index_eye, 36)),
#             ("Высота", self.get_app_data_in(index_eye, 37)),
#             ("Площадь", self.get_app_data_in(index_eye, 38))
#         ]))
#         scroll_layout.addWidget(self.create_block("Жидкость под РПЭ", [
#             ("Пощадь", self.get_app_data_in(index_eye, 39)),
#             ("Локализация", self.get_app_data_in(index_eye, 40))
#         ]))
#         scroll_layout.addWidget(self.create_block("Эллипсоидная зона", [
#             ("Состояние", self.get_app_data_in(index_eye, 41)),
#             ("Локализация дефектов", self.get_app_data_in(index_eye, 42))
#         ]))
#         scroll_layout.addWidget(self.create_block("Миоидная зона", [
#             ("Состояние", self.get_app_data_in(index_eye, 43)),
#             ("Локализация дефектов", self.get_app_data_in(index_eye, 44))
#         ]))
#         scroll_layout.addWidget(self.create_block("Отслойка нейросенсорной сетчатки", [
#             ("Локализация", self.get_app_data_in(index_eye, 45)),
#             ("Ширина", self.get_app_data_in(index_eye, 46)),
#             ("Высота", self.get_app_data_in(index_eye, 47)),
#             ("Площадь", self.get_app_data_in(index_eye, 48))
#         ]))
#         scroll_layout.addWidget(self.create_block("Гиперрефлективный материал", [
#             ("Локализация", self.get_app_data_in(index_eye, 49)),
#             ("Площадь", self.get_app_data_in(index_eye, 50))
#         ]))
#
#         # Устанавливаем макет для контейнера
#         scroll_content.setLayout(scroll_layout)
#         scroll_area.setWidget(scroll_content)
#
#         return scroll_area
#
#     def get_app_data_in(self, x, y):
#         return str(self.appointment_data_model.index(x, y).data())
#     def create_block(self, title, fields):
#         """Создание тематического блока с полями и значениями"""
#         group_box = QGroupBox(title)
#         form_layout = QFormLayout()
#
#         # Добавляем поля с заранее заданными значениями
#         for field, value in fields:
#             input_field = QLineEdit()
#             input_field.setText(value)  # Подстановка значения
#             form_layout.addRow(QLabel(field + ":"), input_field)
#
#         group_box.setLayout(form_layout)
#         return group_box
#
#     def open_dialog(self, patient_id):
#         self.dialog = AddRecordDialog(self.db, patient_id)
#         self.dialog.exec_()  # Запуск модального окна
#
#     def open_dialog_appointmernt(self, patient_id):
#         self.dialog = AddRecordDialog(self.db, patient_id)
#         self.dialog.exec_()  # Запуск модального окна
#
#     def load_appointments_from_db(self, pacientId):
#         db = self.open_connection()
#         if db is None:
#             return  # Завершаем, если подключение не удалось
#
#         self.appointments_model = QSqlQueryModel()
#         query = QSqlQuery()
#         query.prepare("SELECT * FROM appointments WHERE pacient_id = :pacient_id")
#         query.bindValue(":pacient_id", pacientId)
#
#         if query.exec():
#             self.appointments_model.setQuery(query)
#         else:
#             print("Ошибка выполнения запроса:", query.lastError().text())
#
#         self.appointments_table.setModel(self.appointments_model)
#
#         self.appointments_model.insertColumn(self.appointments_model.columnCount())
#
#
#         self.appointments_model.setHeaderData(0, Qt.Orientation.Horizontal, "Ид")
#         self.appointments_model.setHeaderData(1, Qt.Orientation.Horizontal, "Дата приема")
#         self.appointments_model.setHeaderData(2, Qt.Orientation.Horizontal, "Длит. болезни")
#         self.appointments_model.setHeaderData(3, Qt.Orientation.Horizontal, "Редактировать")
#         self.appointments_model.setHeaderData(4, Qt.Orientation.Horizontal, "Показать")
#
#         self.appointments_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
#         self.appointments_table.clicked.connect(self.apointment_clicked)
#
#         self.appointment_edit_delegate2 = EditPacientDelegate(':/dbicons/edit.png', self.patients_table, self.open_dialog_appointmernt)
#         self.appointments_table.setItemDelegateForColumn(self.patients_model.columnCount() - 2, self.appointment_edit_delegate2)
#
#         self.appointment_edit_delegate3 = EditPacientDelegate(':/dbicons/show.png', self.patients_table, self.open_dialog_appointmernt)
#         self.appointments_table.setItemDelegateForColumn(self.patients_model.columnCount() - 1, self.appointment_edit_delegate3)
#
#         self.appointments_table.setColumnHidden(0, True)
#
#     def load_appointment_data_from_db(self, appiontmentId):
#         self.appointment_data_model = QSqlQueryModel()
#         query = QSqlQuery()
#         query.prepare("SELECT * FROM eyes WHERE appointment_id = :appiontmentId")
#         query.bindValue(":appiontmentId", appiontmentId)
#
#         if query.exec():
#             self.appointment_data_model.setQuery(query)
#         else:
#             print("Ошибка выполнения запроса:", query.lastError().text())
#
#         print(self.appointment_data_model.index(0, 0).data())
#
#     def addlay(self):
#         fields = [
#             "id", "eye", "appointment_id", "date", "duration_of_the_disease",
#             "topkon", "optopol", "areds", "refraction", "type_of_neovascularization",
#             "choroidal_thickness_center", "cts_foveola", "cts_sup_inner_fovea",
#             "cts_sup_out_fovea", "total_volume", "average_volume", "rpe_status",
#             "rpe_localisation", "cme_localisation", "serouz_rpe_detachment_localisation",
#             "serouz_rpe_detachment_width", "serouz_rpe_detachment_height", "serouz_rpe_detachment_area",
#             "hemorrhagic_rpe_detachment_localisation", "hemorrhagic_rpe_detachment_width",
#             "hemorrhagic_rpe_detachment_heidgt", "hemorrhagic_rpe_detachment_area",
#             "fibrovascular_rpe_detachment_localisation", "fibrovascular_rpe_detachment_width",
#             "fibrovascular_rpe_detachment_heidgt", "fibrovascular_rpe_detachment_area",
#             "drusenoid_detachment_rpe_localisation", "drusenoid_detachment_rpe_width",
#             "drusenoid_detachment_rpe_height", "drusenoid_detachment_rpe_area",
#             "druses_localisation", "druses_weigt", "druses_heigt", "dzuses_area",
#             "fluid_under_rpe_area", "fluid_under_rpe_localisation", "ez_status",
#             "ez_localisation", "myoidnz_status", "myoidnz_localisation",
#             "rne_detachment_localisation", "rne_detachment_width", "rne_detachment_heigt",
#             "rne_detachment_area", "hyperreflective_material_localisation", "hyperreflective_material_area"
#         ]
#
#         layout = QVBoxLayout()
#
#         # Прокрутка для длинных форм
#         scroll_area = QScrollArea()
#         scroll_widget = QWidget()
#         grid_layout = QGridLayout()
#
#         # Динамическое добавление всех полей
#         row = 0
#         input_fields = {}
#         for field in fields:
#             # Добавляем метку для каждого поля
#             label = QLabel(field.replace('_', ' ').capitalize())
#             grid_layout.addWidget(label, row, 0)  # Первый столбец
#
#             # Добавляем редактируемое поле
#             input_field = QLineEdit()
#             input_fields[field] = input_field
#             grid_layout.addWidget(input_field, row, 1)  # Второй столбец
#
#             row += 1
#
#         # Добавляем разделительную линию
#         separator = QFrame()
#         separator.setFrameShape(QFrame.Shape.HLine)
#         separator.setFrameShadow(QFrame.Shadow.Sunken)
#         grid_layout.addWidget(separator, row, 0, 1, 2)
#
#         row += 1
#
#         # Кнопка сохранения
#         save_button = QPushButton("Сохранить")
#         grid_layout.addWidget(save_button, row, 0, 1, 2)  # На всю ширину
#
#         scroll_widget.setLayout(grid_layout)
#         scroll_area.setWidget(scroll_widget)
#         scroll_area.setWidgetResizable(True)
#
#         layout.addWidget(scroll_area)
#         return layout
#
#
#     def open_connection(self):
#         db = QSqlDatabase.addDatabase("QSQLITE")  # Указываем тип базы данных (SQLite в данном случае)
#         db.setDatabaseName("database.db")  # Указываем имя или путь к базе данных
#
#         if not db.open():  # Проверяем успешность открытия
#             print("Ошибка подключения к базе данных!")
#             return None
#         else:
#             print("Соединение с базой данных установлено.")
#             return db
#
#
# class ButtonDelegate(QStyledItemDelegate):
#     def paint(self, painter, option, index):
#         """Рисуем кнопку в ячейке"""
#         icon_path = ':/dbicons/edit.png'
#         icon = QIcon(icon_path) # 🔹 Здесь нужна иконка (например, карандаш)
#         icon.paint(painter, option.rect)
#
#     def editorEvent(self, event, model, option, index):
#         """Обрабатываем нажатие на кнопку"""
#         if event.type() == event.Type.MouseButtonPress:
#             QMessageBox.information(option.widget, "Редактирование", f"Редактируем строку {index.row()}")
#         return True
#
# class PreviewWindow(QDialog):
#     """Окно для отображения полного изображения."""
#     def __init__(self, image_path):
#         super().__init__()
#         self.setWindowTitle("Полное изображение")
#         self.resize(800, 600)
#
#         layout = QVBoxLayout(self)
#
#         # QLabel для изображения
#         full_image_label = QLabel(self)
#         pixmap = QPixmap(image_path)
#         full_image_label.setPixmap(pixmap)
#         full_image_label.setScaledContents(True)  # Масштабируем изображение
#         layout.addWidget(full_image_label)