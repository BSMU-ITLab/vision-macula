from functools import partial

from PySide6.QtGui import QIcon, Qt, QPixmap
from PySide6.QtSql import QSqlQueryModel, QSqlDatabase, QSqlQuery
from PySide6.QtWidgets import (
    QWidget, QTableWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QLineEdit, QLabel, QScrollArea, QGridLayout,
    QTableView,
    QStyledItemDelegate, QMessageBox, QGroupBox, QFormLayout, QTabWidget, QDialog
)

from bsmu.macula.plugins.db.add_patient_dialog import AddRecordDialog
from bsmu.macula.plugins.db.edit_pacirnt_delegate import EditPacientDelegate
from bsmu.macula.widgets.eye_data_widget import EyeDataWidget
from bsmu.macula.records.eye_info_data import PatientExamData


class TableWidgetExample(QWidget):

    def __init__(self):
        super().__init__()
        self.init_db()
        self.createMainTables()
        self.setWindowTitle("Пациенты")
        self.resize(1200, 800)
        self.appointments_table_created = False
        self.appointment_data_table_created = False

    def init_db(self):
        db = QSqlDatabase.addDatabase("QSQLITE")
        db.setDatabaseName("database.db")
        if not db.open():
            print("Ошибка подключения к базе данных")
            return None
        self.db = db

    def patient_clicked(self, index):
        row = index.row()
        user_id = self.patients_model.index(row, 0).data()
        self.load_appointments_from_db(user_id)

    def apointment_clicked(self, index):
        row = index.row()
        apointment_id = self.appointments_model.index(row, 0).data()
        self.load_appointment_data_from_db(apointment_id)

        self.init_ui()
        # self.main_lay.addLayout(self.init_ui())

    def createMainTables(self):
        self.setWindowTitle("Пациенты")
        self.resize(1200, 800)

        self.main_lay = QHBoxLayout(self)
        left_layout = QVBoxLayout(self)

        db = QSqlDatabase.addDatabase("QSQLITE")
        db.setDatabaseName("database.db")
        if not db.open():
            print("Ошибка подключения к базе данных")
            return None

        self.patients_model = QSqlQueryModel()
        self.patients_model.setQuery("SELECT id, name, sex, year_of_birthday FROM pacients")

        dict_patient = {
            # "id": "Ид",
            "name": "Имя",
            "sex": "Пол",
            "year_of_birthday": "Год рождения",
            "acts": "Действия"
        }

        # for i in range(0, self.patients_model.columnCount()):
        #     self.patients_model.setHeaderData(i, Qt.Orientation.Horizontal, dict_patient[self.patients_model.headerData(i, Orientation.Horizontal )])

        self.patients_table = QTableView(self)
        self.patients_table.setModel(self.patients_model)

        self.patients_model.insertColumn(self.patients_model.columnCount())

        self.patients_model.setHeaderData(0, Qt.Orientation.Horizontal, "Ид")
        self.patients_model.setHeaderData(1, Qt.Orientation.Horizontal, "Имя")
        self.patients_model.setHeaderData(2, Qt.Orientation.Horizontal, "Пол")
        self.patients_model.setHeaderData(3, Qt.Orientation.Horizontal, "Год рождения")
        self.patients_model.setHeaderData(4, Qt.Orientation.Horizontal, "Редактировать")

        self.patients_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.patients_table.clicked.connect(self.patient_clicked)

        self.appointment_edit_delegate = EditPacientDelegate(r"edit.png", self.patients_table, self.open_dialog)
        self.patients_table.setItemDelegateForColumn(self.patients_model.columnCount() - 1, self.appointment_edit_delegate)


        self.patients_table.setColumnHidden(0, True)

        add_patient = QPushButton("Добавить пациента")
        add_patient.clicked.connect(partial(self.open_dialog, 0))
        button_layout = QHBoxLayout()
        button_layout.addWidget(add_patient)

        self.appointments_table = QTableView(self)
        add_appointment_button = QPushButton("Добавить результаты приема")
        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(add_appointment_button)

        left_layout.addWidget(self.patients_table)
        left_layout.addLayout(button_layout)
        left_layout.addWidget(self.appointments_table)
        left_layout.addLayout(button_layout2)
        self.main_lay.addLayout(left_layout)
        layout = QVBoxLayout()
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        self.main_lay.addLayout(layout)
        # layout2 = QVBoxLayout()
        #
        # # QLabel для превью
        # self.preview_label = QLabel("Нажмите на изображение для увеличения")
        # pixmap = QPixmap(rf"C:\Users\Evgeniy\OneDrive\Pictures\32022.jpg")  # Подставьте путь к своему изображению
        # self.preview_label.setPixmap(pixmap.scaled(200, 150))  # Масштабируем превью
        # layout2.addWidget(self.preview_label)
        #
        # # Добавляем обработчик клика
        # self.preview_label.mousePressEvent = self.open_full_image
        # self.main_lay.addLayout(layout2)
        # main_lay.addLayout(self.addlay())

    def open_full_image(self, event):
        """Открывает изображение в отдельном окне."""
        self.dialog = PreviewWindow(rf"C:\Users\Evgeniy\OneDrive\Pictures\32022.jpg")  # Указываем путь к изображению
        self.dialog.exec()

    def init_ui(self):
        if (self.tab_widget.count() > 0) :
            self.tab_widget.removeTab(0)
            self.tab_widget.removeTab(0)
        self.tab_widget.addTab(self.create_scrollable_area("Правый глаз", 0), "Правый глаз")
        self.tab_widget.addTab(self.create_scrollable_area("Левый глаз", 1), "Левый глаз")
        eye_data = PatientExamData.from_query_model(self.appointment_data_model, 1)
        eye_widget = EyeDataWidget(eye_data)
        self.tab_widget.addTab(eye_widget, "Левый глаз")



    def create_scrollable_area(self, layout_name, index_eye):
        """Создаёт область прокрутки с блоками"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Виджет-контейнер для полей
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()

        # Добавляем блоки в прокручиваемую область
        scroll_layout.addWidget(self.create_block("Обследование", [
            ("Дата посещения", self.get_app_data_in(index_eye, 3)),
            ("Продолжительность заболевания", self.get_app_data_in(index_eye, 4)),
            ("Тип томографа ","Топкон"),
            ("Критерий AREDS", self.get_app_data_in(index_eye, 7)),
            ("Рефракция", self.get_app_data_in(index_eye, 8)),
            ("Тип неоваскуляризации", self.get_app_data_in(index_eye, 9))
        ]))
        scroll_layout.addWidget(self.create_block("Ретинальные показатели", [
            ("Толщина хориоидеи в центре", self.get_app_data_in(index_eye, 10)),
            ("Толщина сетчатки в фовеоле", self.get_app_data_in(index_eye, 11)),
            ("Общий объем", self.get_app_data_in(index_eye, 14)),
            ("Средний объем", self.get_app_data_in(index_eye, 15))
        ]))
        scroll_layout.addWidget(self.create_block("", [
            ("Состояние РПЭ", self.get_app_data_in(index_eye, 16)),
            ("Локализация дефектов РПЭ", self.get_app_data_in(index_eye, 17)),
            ("Локализация кистозного макулярного отека", self.get_app_data_in(index_eye, 18))
        ]))
        scroll_layout.addWidget(QLabel("Отслойки РПЭ"))
        scroll_layout.addWidget(self.create_block("Серозная ОПЭ", [
            ("Локализация", self.get_app_data_in(index_eye, 19)),
            ("Ширина", self.get_app_data_in(index_eye, 20)),
            ("Высота", self.get_app_data_in(index_eye, 21)),
            ("Площадь", self.get_app_data_in(index_eye, 22))
        ]))
        scroll_layout.addWidget(self.create_block("Геморрагическая ОПЭ", [
            ("Локализация", self.get_app_data_in(index_eye, 23)),
            ("Ширина", self.get_app_data_in(index_eye, 24)),
            ("Высота", self.get_app_data_in(index_eye, 25)),
            ("Площадь", self.get_app_data_in(index_eye, 26))
        ]))
        scroll_layout.addWidget(self.create_block("Фиброваскулярная ОПЭ", [
            ("Локализация", self.get_app_data_in(index_eye, 27)),
            ("Ширина", self.get_app_data_in(index_eye, 28)),
            ("Высота", self.get_app_data_in(index_eye, 29)),
            ("Площадь", self.get_app_data_in(index_eye, 30))
        ]))
        scroll_layout.addWidget(self.create_block("Друзеноидная ОПЭ", [
            ("Локализация", self.get_app_data_in(index_eye, 31)),
            ("Ширина", self.get_app_data_in(index_eye, 32)),
            ("Высота", self.get_app_data_in(index_eye, 33)),
            ("Площадь", self.get_app_data_in(index_eye, 34))
        ]))
        scroll_layout.addWidget(self.create_block("Друзы", [
            ("Локализация", self.get_app_data_in(index_eye, 35)),
            ("Ширина", self.get_app_data_in(index_eye, 36)),
            ("Высота", self.get_app_data_in(index_eye, 37)),
            ("Площадь", self.get_app_data_in(index_eye, 38))
        ]))
        scroll_layout.addWidget(self.create_block("Жидкость под РПЭ", [
            ("Пощадь", self.get_app_data_in(index_eye, 39)),
            ("Локализация", self.get_app_data_in(index_eye, 40))
        ]))
        scroll_layout.addWidget(self.create_block("Эллипсоидная зона", [
            ("Состояние", self.get_app_data_in(index_eye, 41)),
            ("Локализация дефектов", self.get_app_data_in(index_eye, 42))
        ]))
        scroll_layout.addWidget(self.create_block("Миоидная зона", [
            ("Состояние", self.get_app_data_in(index_eye, 43)),
            ("Локализация дефектов", self.get_app_data_in(index_eye, 44))
        ]))
        scroll_layout.addWidget(self.create_block("Отслойка нейросенсорной сетчатки", [
            ("Локализация", self.get_app_data_in(index_eye, 45)),
            ("Ширина", self.get_app_data_in(index_eye, 46)),
            ("Высота", self.get_app_data_in(index_eye, 47)),
            ("Площадь", self.get_app_data_in(index_eye, 48))
        ]))
        scroll_layout.addWidget(self.create_block("Гиперрефлективный материал", [
            ("Локализация", self.get_app_data_in(index_eye, 49)),
            ("Площадь", self.get_app_data_in(index_eye, 50))
        ]))

        # Устанавливаем макет для контейнера
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)

        return scroll_area

    def get_app_data_in(self, x, y):
        return str(self.appointment_data_model.index(x, y).data())
    def create_block(self, title, fields):
        """Создание тематического блока с полями и значениями"""
        group_box = QGroupBox(title)
        form_layout = QFormLayout()

        # Добавляем поля с заранее заданными значениями
        for field, value in fields:
            input_field = QLineEdit()
            input_field.setText(value)  # Подстановка значения
            form_layout.addRow(QLabel(field + ":"), input_field)

        group_box.setLayout(form_layout)
        return group_box

    def open_dialog(self, patient_id):
        self.dialog = AddRecordDialog(self.db, patient_id)
        self.dialog.exec_()  # Запуск модального окна

    def open_dialog_appointmernt(self, patient_id):
        self.dialog = AddRecordDialog(self.db, patient_id)
        self.dialog.exec_()  # Запуск модального окна

    def load_appointments_from_db(self, pacientId):
        db = self.open_connection()
        if db is None:
            return  # Завершаем, если подключение не удалось

        self.appointments_model = QSqlQueryModel()
        query = QSqlQuery()
        query.prepare("SELECT * FROM appointments WHERE pacient_id = :pacient_id")
        query.bindValue(":pacient_id", pacientId)

        if query.exec():
            self.appointments_model.setQuery(query)
        else:
            print("Ошибка выполнения запроса:", query.lastError().text())

        table1Description = ["Ид", "Дата приема", "duration_of_the_disease", "Действия"]


        # for col, field in enumerate(table1Description.keys()):
        #     self.patients_model.setHeaderData(col, Qt.Orientation.Horizontal, table1Description[field])

        self.appointments_table.setModel(self.appointments_model)

        self.appointments_model.insertColumn(self.appointments_model.columnCount())


        self.appointments_model.setHeaderData(0, Qt.Orientation.Horizontal, "Ид")
        self.appointments_model.setHeaderData(1, Qt.Orientation.Horizontal, "Дата приема")
        self.appointments_model.setHeaderData(2, Qt.Orientation.Horizontal, "Длит. болезни")
        # self.appointments_model.setHeaderData(3, Qt.Orientation.Horizontal, "Ид пациента")
        self.appointments_model.setHeaderData(3, Qt.Orientation.Horizontal, "Редактировать")
        self.appointments_model.setHeaderData(4, Qt.Orientation.Horizontal, "Показать")

        self.appointments_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.appointments_table.clicked.connect(self.apointment_clicked)

        self.appointment_edit_delegate2 = EditPacientDelegate(r"edit.png", self.patients_table, self.open_dialog_appointmernt)
        self.appointments_table.setItemDelegateForColumn(self.patients_model.columnCount() - 2, self.appointment_edit_delegate2)

        self.appointment_edit_delegate3 = EditPacientDelegate(r"2.png", self.patients_table, self.open_dialog_appointmernt)
        self.appointments_table.setItemDelegateForColumn(self.patients_model.columnCount() - 1, self.appointment_edit_delegate3)

        self.appointments_table.setColumnHidden(0, True)

    def load_appointment_data_from_db(self, appiontmentId):
        self.appointment_data_model = QSqlQueryModel()
        query = QSqlQuery()
        query.prepare("SELECT * FROM eyes WHERE appointment_id = :appiontmentId")
        query.bindValue(":appiontmentId", appiontmentId)

        if query.exec():
            self.appointment_data_model.setQuery(query)
        else:
            print("Ошибка выполнения запроса:", query.lastError().text())

        print(self.appointment_data_model.index(0, 0).data())

    def addlay(self):
        fields = [
            "id", "eye", "appointment_id", "date", "duration_of_the_disease",
            "topkon", "optopol", "areds", "refraction", "type_of_neovascularization",
            "choroidal_thickness_center", "cts_foveola", "cts_sup_inner_fovea",
            "cts_sup_out_fovea", "total_volume", "average_volume", "rpe_status",
            "rpe_localisation", "cme_localisation", "serouz_rpe_detachment_localisation",
            "serouz_rpe_detachment_width", "serouz_rpe_detachment_height", "serouz_rpe_detachment_area",
            "hemorrhagic_rpe_detachment_localisation", "hemorrhagic_rpe_detachment_width",
            "hemorrhagic_rpe_detachment_heidgt", "hemorrhagic_rpe_detachment_area",
            "fibrovascular_rpe_detachment_localisation", "fibrovascular_rpe_detachment_width",
            "fibrovascular_rpe_detachment_heidgt", "fibrovascular_rpe_detachment_area",
            "drusenoid_detachment_rpe_localisation", "drusenoid_detachment_rpe_width",
            "drusenoid_detachment_rpe_height", "drusenoid_detachment_rpe_area",
            "druses_localisation", "druses_weigt", "druses_heigt", "dzuses_area",
            "fluid_under_rpe_area", "fluid_under_rpe_localisation", "ez_status",
            "ez_localisation", "myoidnz_status", "myoidnz_localisation",
            "rne_detachment_localisation", "rne_detachment_width", "rne_detachment_heigt",
            "rne_detachment_area", "hyperreflective_material_localisation", "hyperreflective_material_area"
        ]

        layout = QVBoxLayout()

        # Прокрутка для длинных форм
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        grid_layout = QGridLayout()

        # Динамическое добавление всех полей
        row = 0
        input_fields = {}
        for field in fields:
            # Добавляем метку для каждого поля
            label = QLabel(field.replace('_', ' ').capitalize())
            grid_layout.addWidget(label, row, 0)  # Первый столбец

            # Добавляем редактируемое поле
            input_field = QLineEdit()
            input_fields[field] = input_field
            grid_layout.addWidget(input_field, row, 1)  # Второй столбец

            row += 1

        # Добавляем разделительную линию
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        grid_layout.addWidget(separator, row, 0, 1, 2)

        row += 1

        # Кнопка сохранения
        save_button = QPushButton("Сохранить")
        grid_layout.addWidget(save_button, row, 0, 1, 2)  # На всю ширину

        scroll_widget.setLayout(grid_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        layout.addWidget(scroll_area)
        return layout


    def open_connection(self):
        db = QSqlDatabase.addDatabase("QSQLITE")  # Указываем тип базы данных (SQLite в данном случае)
        db.setDatabaseName("database.db")  # Указываем имя или путь к базе данных

        if not db.open():  # Проверяем успешность открытия
            print("Ошибка подключения к базе данных!")
            return None
        else:
            print("Соединение с базой данных установлено.")
            return db


class ButtonDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        """Рисуем кнопку в ячейке"""
        icon_path = r"/bsmu/macula/app/images/icons/edit.png"
        icon = QIcon(icon_path) # 🔹 Здесь нужна иконка (например, карандаш)
        icon.paint(painter, option.rect)

    def editorEvent(self, event, model, option, index):
        """Обрабатываем нажатие на кнопку"""
        if event.type() == event.Type.MouseButtonPress:
            QMessageBox.information(option.widget, "Редактирование", f"Редактируем строку {index.row()}")
        return True

    # def init_ui(self):
    #     # Основной виджет
    #     layout = QVBoxLayout()
    #
    #     # Добавляем блоки с прокруткой
    #     layout.addWidget(self.create_scrollable_area())
    #
    #     # Устанавливаем главный макет
    #     return layout
    #
    # def create_scrollable_area(self):
    #     """Создаёт область прокрутки с блоками"""
    #     scroll_area = QScrollArea()
    #     scroll_area.setWidgetResizable(True)
    #
    #     # Виджет-контейнер для полей
    #     scroll_content = QWidget()
    #     scroll_layout = QVBoxLayout()
    #
    #     # Добавляем блоки в прокручиваемую область
    #     scroll_layout.addWidget(self.create_block("Ид и основное", [
    #         "Ид", "Глаз", "Ид посещения", "Дата посещения", "Продолжительность заболевания"
    #     ]))
    #     scroll_layout.addWidget(self.create_block("Обследования", [
    #         "топкон", "оптопол", "критерий AREDS", "рефракция", "тип неоваскуляризации"
    #     ]))
    #     scroll_layout.addWidget(self.create_block("Хороидальная толщина", [
    #         "толщина хориоидеи в центре", "толщина ЦТС", "толщина ЦТС верхнего внутреннего слоя ямки",
    #         "толщина ЦТС верхнего наружного слоя ямки", "общий объем", "средний объем"
    #     ]))
    #     scroll_layout.addWidget(self.create_block("Состояние ЭПС", [
    #         "состояние ЭПС", "локализация ЭПС", "локализация ЦМЭ",
    #         "локализация серозного отслоения ЭПС", "ширина серозного отслоения ЭПС",
    #         "высота серозного отслоения ЭПС", "площадь серозного отслоения ЭПС"
    #     ]))
    #     scroll_layout.addWidget(self.create_block("Геморрагическое ЭПС", [
    #         "локализация геморрагического отслоения ЭПС", "ширина геморрагического отслоения ЭПС",
    #         "высота геморрагического отслоения ЭПС", "площадь геморрагического отслоения ЭПС"
    #     ]))
    #     scroll_layout.addWidget(self.create_block("Фиброваскулярное ЭПС", [
    #         "локализация фиброваскулярного отслоения ЭПС", "ширина фиброваскулярного отслоения ЭПС",
    #         "высота фиброваскулярного отслоения ЭПС", "площадь фиброваскулярного отслоения ЭПС"
    #     ]))
    #     scroll_layout.addWidget(self.create_block("Друзы и ЭПС", [
    #         "локализация друзеноидного отслоения ЭПС", "ширина друзеноидного отслоения ЭПС",
    #         "высота друзеноидного отслоения ЭПС", "площадь друзеноидного отслоения ЭПС",
    #         "локализация друз", "вес друз", "высота друз", "площадь друз"
    #     ]))
    #     scroll_layout.addWidget(self.create_block("Жидкость", [
    #         "площадь жидкости под ЭПС", "локализация жидкости под ЭПС"
    #     ]))
    #     scroll_layout.addWidget(self.create_block("зоны Эллера и прочее", [
    #         "состояние EZ", "локализация EZ", "состояние миоидной зоны", "локализация миоидной зоны"
    #     ]))
    #     scroll_layout.addWidget(self.create_block("сетчатки и гиперрефлективный материал", [
    #         "локализация отслоения РНЭ", "ширина отслоения РНЭ", "высота отслоения РНЭ",
    #         "площадь отслоения РНЭ", "локализация гиперотражающего материала",
    #         "площадь гиперотражающего материала"
    #     ]))
    #
    #     # Устанавливаем макет для виджета
    #     scroll_content.setLayout(scroll_layout)
    #     scroll_area.setWidget(scroll_content)
    #
    #     return scroll_area
    #
    # def create_block(self, title, fields):
    #     """Создание тематического блока с полями"""
    #     group_box = QGroupBox(title)
    #     form_layout = QFormLayout()
    #
    #     # Добавляем поля
    #     for field in fields:
    #         form_layout.addRow(QLabel(field + ":"), QLineEdit())
    #
    #     group_box.setLayout(form_layout)
    #     return group_box
class PreviewWindow(QDialog):
    """Окно для отображения полного изображения."""
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Полное изображение")
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        # QLabel для изображения
        full_image_label = QLabel(self)
        pixmap = QPixmap(image_path)
        full_image_label.setPixmap(pixmap)
        full_image_label.setScaledContents(True)  # Масштабируем изображение
        layout.addWidget(full_image_label)