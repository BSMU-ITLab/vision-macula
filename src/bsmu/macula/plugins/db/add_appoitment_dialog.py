from functools import partial

from PySide6.QtCore import QDate, Signal
from PySide6.QtSql import QSqlQuery
from PySide6.QtWidgets import QDialog, QLineEdit, QPushButton, QVBoxLayout, QLabel, QDateEdit, QRadioButton, QGroupBox

from bsmu.macula.plugins.db.database_manager import DatabaseManager

class AddAppoitmentRecordDialog(QDialog):

    dataReady = Signal(dict)
    def __init__(self, db, patient_id):
        super().__init__()
        self.db = db
        self.eyes = ['L', 'R']
        self.last_id = None
        self.db_manager = DatabaseManager()
        self.setWindowTitle("Добавить прием")
        self.pat_id = patient_id

        self.time_input = QLineEdit()
        self.save_button = QPushButton("Сохранить")
        self.save_button.clicked.connect(self.save_data)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Дата приема:"))
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        layout.addWidget(self.date_edit)
        layout.addWidget(QLabel("Длительность заболевания:"))
        layout.addWidget(self.time_input)

        # --- Основные радиокнопки ---
        self.radio_one = QRadioButton("Записать данные по одному глазу")
        self.radio_two = QRadioButton("Записать данные по обоим глазам")
        layout.addWidget(self.radio_one)
        layout.addWidget(self.radio_two)

        # --- Группа для уточнения (правый/левый) ---
        self.eye_choice_group = QGroupBox("Выберите глаз")
        eye_layout = QVBoxLayout()

        self.radio_right = QRadioButton("Правый глаз")
        self.radio_left = QRadioButton("Левый глаз")

        eye_layout.addWidget(self.radio_right)
        eye_layout.addWidget(self.radio_left)

        self.eye_choice_group.setLayout(eye_layout)
        layout.addWidget(self.eye_choice_group)

        # --- Логика переключения ---
        self.radio_one.toggled.connect(self.update_visibility)
        self.radio_two.toggled.connect(self.update_visibility)

        self.update_visibility()  # начальное состояние

        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def get_eyes_info(self):
        return self.eyes

    def update_visibility(self):
        if self.radio_one.isChecked():
            self.eye_choice_group.show()  # показываем уточнение
        else:
            self.eye_choice_group.hide()  # скрываем уточнение

    def save_data(self):
        date = self.date_edit.text()
        time = self.time_input.text()
        if self.radio_right.isChecked():
            self.eyes = ['R']
        if self.radio_left.isChecked():
            self.eyes = ['L']
        if date and time:
            query = QSqlQuery(self.db)
            query.prepare("INSERT INTO appointments (date, duration_of_the_disease, pacient_id) VALUES (?, ?, ?)")
            query.addBindValue(date)
            query.addBindValue(time)
            query.addBindValue(self.pat_id)
            if query.exec_():
                # Получаем последний вставленный ID
                self.last_id = query.lastInsertId()
                print("Создана запись с ID:", self.last_id)
            else:
                print("Ошибка:", query.lastError().text())
            self.close()
        else:
            print("Ошибка: введите корректные данные")

    def closeEvent(self, event):
        dat = {'e': self.eyes, 'ap_id': self.last_id}
        self.dataReady.emit(dat)

        super().closeEvent(event)