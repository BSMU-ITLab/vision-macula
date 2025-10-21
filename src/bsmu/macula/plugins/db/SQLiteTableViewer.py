from functools import partial

from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlQueryModel, QSqlQuery

from bsmu.macula.plugins.db.DynamicFormBuilder import DynamicFormBuilder
from bsmu.macula.plugins.db.HoverComboBox import HoverComboBox
from bsmu.macula.plugins.db.PatientsModel import PatientsModel
from bsmu.macula.plugins.db.constants import DROPDOWN_DB_VALUES, DROPDOWN_DISPLAY_MAP, BLOCKS, \
    COLUMN_INDEX_TO_FIELD_NAME, COLUMNS_EYES, EYE_TABLE_NAME, ERROR_TEXT, PACIENTS_TABLE_NAME, \
    PARAMETERS_PATIENTS_SELECT
from bsmu.macula.plugins.db.debug_utils import print_sql_debug
from bsmu.macula.plugins.db.edit_pacirnt_delegate import EditPacientDelegate
from PySide6.QtWidgets import (
    QWidget, QTableView, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QGroupBox, QLineEdit,
    QTabWidget, QMessageBox, QComboBox
)
import bsmu.macula.plugins.db.images.dbicons_rc

from bsmu.macula.plugins.db.add_patient_dialog import AddRecordDialog
from bsmu.macula.plugins.db.database_manager_v2 import DatabaseManager
from bsmu.macula.plugins.db.query_builder import QueryBuilder


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
            self.query_builder_eye = QueryBuilder(EYE_TABLE_NAME, self.db_manager)
            self.query_builder_patient = QueryBuilder(PACIENTS_TABLE_NAME, self.db_manager)
            self.layoutBoxes = QVBoxLayout()
            self._setup_ui()
            self._load_initial_data()
        except ConnectionError as e:
            QMessageBox.critical(self, ERROR_TEXT, str(e))
            self.close()

    def _setup_ui(self):
        # self.setWindowTitle("Подсказка при наведении (PySide6)")
        #
        # layout = QVBoxLayout()
        # self.combo = HoverComboBox()
        # self.combo.addItems(["Опция 1", "Опция 2", "Опция 3", "Опция 4"])
        #
        # layout.addWidget(self.combo)
        #
        # self.setLayout(layout)
        self.appointment_data_model = QSqlQueryModel()
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
        self.patients_model.setQuery(self._create_load_patients_query())

        headers = ["Ид", "Имя", "Пол", "Год рождения", "Редактировать"]
        for i, header in enumerate(headers):
            self.patients_model.setHeaderData(i, Qt.Horizontal, header)

        self.patients_table.setModel(self.patients_model)
        self.patients_table.setSelectionBehavior(QTableView.SelectRows)
        self.patients_table.clicked.connect(self._on_patient_clicked)
        self.patients_table.setColumnHidden(0, True)
        self.appointment_edit_delegate = EditPacientDelegate(':/dbicons/edit.png', self.patients_table, self._open_dialog)
        self.patients_table.setItemDelegateForColumn(self.patients_model.columnCount() - 1, self.appointment_edit_delegate)

        add_button = QPushButton("Добавить пациента")
        add_button.clicked.connect(partial(self._open_dialog, 0))

        self.left_layout.addWidget(self.patients_table)
        self.left_layout.addWidget(add_button)

    def _create_load_patients_query(self):
        patients_query = self.query_builder_patient.select(PARAMETERS_PATIENTS_SELECT)
        patients_query.exec_()

        return patients_query
    def _load_patients_data(self):
        self.patients_model.setQuery(self._create_load_patients_query())
        self.patients_table.setModel(self.patients_model)

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
        columns = COLUMNS_EYES + [
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
            "injections.lucentis_injections"
        ]

        # Формируем SQL-запрос
        column_clause = ", ".join([f"eyes.{col}" if not col.startswith("injections.") else col for col in columns])

        query_text = f"""
        SELECT {column_clause}
        FROM eyes
        LEFT JOIN injections ON injections.eye_id = eyes.id
        WHERE eyes.appointment_id = {appointment_id}
        """

        # Выполняем запрос
        model = QSqlQueryModel()
        query = QSqlQuery(self.db_manager.get_connection())
        t = query.exec_(query_text)
        model.setQuery(query)
        self.appointment_id = appointment_id
        self.appointment_data_model = model
        self.print_sql_model_data(model)


    def get_value_by_field(self, model, row_index, field_name):
        # Получаем индекс столбца по имени
        column_index = None
        for col in range(model.columnCount()):
            header = model.headerData(col, Qt.Orientation.Horizontal)
            if header == field_name:
                column_index = col
                break
        if column_index is None:
            raise ValueError(f"Поле '{field_name}' не найдено в модели.")

        index = model.index(row_index, column_index)
        return model.data(index)

    def _update_ui(self, eye_id_field):
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
        rowIntL = None
        rowIntR = None
        rowInt = 0
        if eye_id_field is not None :
            for row in range(self.appointment_data_model.rowCount()):
                index = self.appointment_data_model.index(row, 0)  # Предположим, что id в первом столбце
                id_value = self.appointment_data_model.data(index)
                eye_in = self.appointment_data_model.data(self.appointment_data_model.index(row, 1))
                if eye_in == 'L' or eye_in == '0' or eye_in == 'ос':
                    lId = id_value
                    rowIntL = rowInt
                if eye_in == 'R' or eye_in == '1' or eye_in == 'од':
                    rId = id_value
                    rowIntR = rowInt
                ids.append(id_value)
                rowInt = rowInt + 1


        # Добавление вкладок для каждого глаза
        self.tab_widget.addTab(self._create_eye_tab(rowIntR, rId), "Правый глаз")
        self.tab_widget.addTab(self._create_eye_tab(rowIntL, lId), "Левый глаз")

    def print_sql_model_data(self, model):
        rows = model.rowCount()
        cols = model.columnCount()
        for row in range(rows):
            values = []
            for col in range(cols):
                index = model.index(row, col)
                values.append(str(model.data(index)))
            print(f"Row {row}: {values}")

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

    def collect_form_data(self, form_dict, eye_index: int, blocks: list, get_column_name_fn) -> tuple[list, dict]:
        all_fields = []
        all_values = {}

        for block_title, fields in blocks:
            for label, field_name2 in fields:
                field_name = get_column_name_fn(field_name2)
                widget_key = field_name + str(eye_index)
                widget = form_dict[eye_index].get_all_field_widgets().get(widget_key)

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

        return all_fields, all_values

    def _toggle_edit_mode_2(self, button: QPushButton, eye_index: int):
        """Переключает режим редактирования всех блоков и вставляет новую запись"""
        # Переключаем все поля
        table_name = "eyes"
        all_fields, all_values = self.collect_form_data(
            self.formDict,
            eye_index,
            BLOCKS,
            self._get_column_name
        )

        # Проверка на наличие данных
        if not all_fields:
            print("❌ Нет данных для вставки.")
            return

        if (self.appointment_id is None or self.appointment_id == 0):
            # Вставка в appointmentsw
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
        all_fields += ["appointment_id", "eye", "topkon", "optopol"]
        all_values["appointment_id"] = self.appointment_id
        all_values["eye"] = eye_index
        # all_values["MKO"] = eye_index
        all_values["optopol"] = True

        # Вставка
        # query = self.builder.insert(all_values)
        # s = query.exec_()
        # Формируем SQL-запрос
        field_clause = ", ".join(all_fields)
        placeholder_clause = ", ".join([f":{field}" for field in all_fields])
        query_text = f"INSERT INTO {table_name} ({field_clause}) VALUES ({placeholder_clause})"

        query2 = QSqlQuery(self.db_manager.get_connection())
        query2.prepare(query_text)

        for field, value in all_values.items():
            query2.bindValue(f":{field}", value)
        print_sql_debug(query_text, query2, all_fields, '1', '1')
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
