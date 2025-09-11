from functools import partial

from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlQueryModel, QSqlQuery

from bsmu.macula.plugins.db.DynamicFormBuilder import DynamicFormBuilder
from bsmu.macula.plugins.db.PatientsModel import PatientsModel
from bsmu.macula.plugins.db.constants import DROPDOWN_DB_VALUES, DROPDOWN_DISPLAY_MAP, BLOCKS, \
    COLUMN_INDEX_TO_FIELD_NAME
from bsmu.macula.plugins.db.edit_pacirnt_delegate import EditPacientDelegate
from PySide6.QtWidgets import (
    QWidget, QTableView, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QGroupBox, QFormLayout, QLineEdit,
    QTabWidget, QMessageBox, QComboBox
)
import bsmu.macula.plugins.db.images.dbicons_rc

from bsmu.macula.plugins.db.add_patient_dialog import AddRecordDialog
from bsmu.macula.plugins.db.database_manager_v2 import DatabaseManager


class TableWidgetExample(QWidget):



    def __init__(self, bd_path):
        super().__init__()
        self.formDict = {}
        self.dropdown_db_values = DROPDOWN_DB_VALUES

        self.dropdown_display_map = DROPDOWN_DISPLAY_MAP
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

        blocks = BLOCKS
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
        column_index_to_field_name = COLUMN_INDEX_TO_FIELD_NAME

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

    def _get_column_name(self, column_index: int) -> str:
        """Возвращает имя поля по индексу"""
        column_index_to_field_name = COLUMN_INDEX_TO_FIELD_NAME

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
