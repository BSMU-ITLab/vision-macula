import os
from functools import partial

from PySide6.QtSql import QSqlQuery
from PySide6.QtWidgets import QDialog, QLineEdit, QPushButton, QVBoxLayout, QLabel

from bsmu.macula.plugins.db.database_manager import DatabaseManager


class AddRecordDialog(QDialog):
    def __init__(self, db, patient_id):
        super().__init__()
        self.db = db
        self.db_manager = DatabaseManager()
        self.is_new_user = patient_id == 0
        self.setWindowTitle("Добавить пациента" if self.is_new_user else "Изменить данные пациента")

        self.name_input = QLineEdit()
        self.sex_input = QLineEdit()
        self.age_input = QLineEdit()
        self.save_button = QPushButton("Сохранить" if self.is_new_user else "Редактировать")
        if (self.is_new_user):
            self.save_button.clicked.connect(self.save_data)
        else:
            self.save_button.clicked.connect(partial(self.edit_data, patient_id))

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Имя:"))
        layout.addWidget(self.name_input)
        layout.addWidget(QLabel("Пол:"))
        layout.addWidget(self.sex_input)
        layout.addWidget(QLabel("Год рождения:"))
        layout.addWidget(self.age_input)
        layout.addWidget(self.save_button)

        if (not self.is_new_user):
            record = self.db_manager.fetch_record_by_id("pacients", patient_id)
            # query = QSqlQuery(self.db)
            # query.prepare("SELECT id, name, sex, year_of_birthday FROM pacients WHERE id = :pacient_id")
            # query.bindValue(":pacient_id", patient_id)
            # query.exec_()
            # self.appointments_model = QSqlQueryModel()
            # self.appointments_model.setQuery(query)
            self.name_input.setText(record[0][1])
            self.sex_input.setText(record[0][2])
            self.age_input.setText(str(record[0][3]))
        self.setLayout(layout)

    def save_data(self):
        name = self.name_input.text()
        sex = self.sex_input.text()
        age = self.age_input.text()

        if name and age.isdigit():
            query = QSqlQuery(self.db)
            query.prepare("INSERT INTO pacients (name, sex, year_of_birthday) VALUES (?, ?, ?)")
            query.addBindValue(name)
            query.addBindValue(sex)
            query.addBindValue(int(age))
            query.exec_()
            last_id = query.lastInsertId()
            self.create_forder(last_id)
            self.close()
        else:
            print("Ошибка: введите корректные данные")

    def create_forder(self, last_id):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Относительный путь (например, "data" внутри папки программы)
        folder = os.path.join(base_dir, "data", str(last_id))

        # Создаём папку, если её нет
        os.makedirs(folder, exist_ok=True)

        print(f"Папка создана: {folder}")
    def edit_data(self, patient_id):
        name = self.name_input.text()
        sex = self.sex_input.text()
        age = self.age_input.text()

        if name and sex and age.isdigit():
            self.db_manager.update_pacient(name, sex, int(age), patient_id)
            # query = QSqlQuery(self.db)
            # query.prepare("UPDATE pacients SET name = ? WHERE id = ?")
            # query.addBindValue(name)
            # query.addBindValue(sex)
            # query.addBindValue(int(age))
            # query.addBindValue(patient_id)
            # bol = query.exec_()
            self.close()
        else:
            print("Ошибка: введите корректные данные")